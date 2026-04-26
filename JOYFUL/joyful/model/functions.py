import numpy as np
import torch
import torch.nn.functional as F

import joyful

log = joyful.utils.get_logger()


def batch_graphify(features, lengths, speaker_tensor, wp, wf, edge_type_to_idx, device):
    node_features, edge_index, edge_type = [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j].cpu().item(), wp, wf))

    for j in range(batch_size):
        cur_len = lengths[j].item()
        node_features.append(features[j, :cur_len, :])
        perms = edge_perms(cur_len, wp, wf)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        length_sum += cur_len
        edge_index_lengths.append(len(perms))
        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))

            speaker1 = speaker_tensor[j, item[0]].item()
            speaker2 = speaker_tensor[j, item[1]].item()
            if item[0] < item[1]:
                c = "0"
            else:
                c = "1"
            edge_type.append(edge_type_to_idx[str(speaker1) + str(speaker2) + c])

    node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]
    edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
    edge_type = torch.tensor(edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)  # [B]

    return node_features, edge_index, edge_type, edge_index_lengths


def edge_perms(length, window_past, window_future):
    """
    Method to construct the edges of a graph (a utterance) considering the past and future window.
    return: list of tuples. tuple -> (vertice(int), neighbor(int))
    """

    all_perms = set()
    array = np.arange(length)
    for j in range(length):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:  # use all past context
            eff_array = array[: min(length, j + window_future + 1)]
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past) :]
        else:
            eff_array = array[
                max(0, j - window_past) : min(length, j + window_future + 1)
            ]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)


def compute_pair_similarity(node_feats, metric="cosine"):
    if metric == "dot":
        return torch.matmul(node_feats, node_feats.t())
    z = F.normalize(node_feats, dim=-1, p=2)
    return torch.matmul(z, z.t())


def build_similarity_hyperedges(
    sim,
    threshold=0.7,
    topk=-1,
    min_size=3,
    max_size=8,
    max_hyperedges=30,
):
    length = sim.size(0)
    hyperedges = []
    seen = set()
    for i in range(length):
        row = sim[i].clone()
        row[i] = -1e9
        if topk is not None and topk > 0:
            k = min(topk, max(0, length - 1))
            _, idxs = torch.topk(row, k=k)
            nbrs = [j.item() for j in idxs if row[j] >= threshold]
        else:
            nbrs = [j for j in range(length) if row[j] >= threshold]
        nodes = [i] + nbrs
        if len(nodes) < min_size:
            continue
        if len(nodes) > max_size:
            nodes = nodes[:max_size]
        key = tuple(sorted(nodes))
        if key in seen:
            continue
        seen.add(key)
        hyperedges.append(list(key))
        if len(hyperedges) >= max_hyperedges:
            break
    return hyperedges


def expand_hyperedges_to_binary(hyperedges):
    edges = set()
    for he in hyperedges:
        if len(he) < 2:
            continue
        for i in range(len(he)):
            for j in range(len(he)):
                if i != j:
                    edges.add((he[i], he[j]))
    return edges


def batch_hybrid_graphify(
    features, lengths, speaker_tensor, wp, wf, edge_type_to_idx, device, args
):
    node_features, edge_index, edge_type = [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_index_lengths = []
    hyper_rel_idx = edge_type_to_idx.get("HYPER", None)
    sim_metric = getattr(args, "sim_metric", "cosine")
    sim_threshold = getattr(args, "sim_threshold", 0.7)
    sim_topk = getattr(args, "sim_topk", -1)
    hyper_min_size = getattr(args, "hyper_min_size", 3)
    hyper_max_size = getattr(args, "hyper_max_size", 8)
    max_hyperedges = getattr(args, "max_hyperedges_per_dialog", 30)
    edge_ratio_cap = getattr(args, "hyper_edge_ratio_cap", 1.0)

    for b in range(batch_size):
        cur_len = lengths[b].item()
        local_feats = features[b, :cur_len, :]
        node_features.append(local_feats)

        binary_edges = set(edge_perms(cur_len, wp, wf))
        sim = compute_pair_similarity(local_feats, metric=sim_metric)
        hyperedges = build_similarity_hyperedges(
            sim,
            threshold=sim_threshold,
            topk=sim_topk,
            min_size=hyper_min_size,
            max_size=hyper_max_size,
            max_hyperedges=max_hyperedges,
        )
        expanded_edges = list(expand_hyperedges_to_binary(hyperedges))
        if edge_ratio_cap > 0:
            max_added = int(len(binary_edges) * edge_ratio_cap)
            expanded_edges = expanded_edges[:max_added]

        merged_edges = []
        merged_types = []
        for src, dst in binary_edges:
            merged_edges.append((src, dst))
            speaker1 = speaker_tensor[b, src].item()
            speaker2 = speaker_tensor[b, dst].item()
            c = "0" if src < dst else "1"
            merged_types.append(edge_type_to_idx[str(speaker1) + str(speaker2) + c])

        for src, dst in expanded_edges:
            if (src, dst) in binary_edges:
                continue
            merged_edges.append((src, dst))
            if hyper_rel_idx is None:
                speaker1 = speaker_tensor[b, src].item()
                speaker2 = speaker_tensor[b, dst].item()
                c = "0" if src < dst else "1"
                merged_types.append(edge_type_to_idx[str(speaker1) + str(speaker2) + c])
            else:
                merged_types.append(hyper_rel_idx)

        edge_index_lengths.append(len(merged_edges))
        for (src, dst), e_t in zip(merged_edges, merged_types):
            edge_index.append(torch.tensor([src + length_sum, dst + length_sum]))
            edge_type.append(e_t)
        length_sum += cur_len

    node_features = torch.cat(node_features, dim=0).to(device)
    edge_index = torch.stack(edge_index).t().contiguous().to(device)
    edge_type = torch.tensor(edge_type).long().to(device)
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)

    return node_features, edge_index, edge_type, edge_index_lengths
