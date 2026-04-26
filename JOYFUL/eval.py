import pickle
import argparse
import os
import torch
from sklearn import metrics
from tqdm import tqdm
import joyful

log = joyful.utils.get_logger()


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def resolve_data_path(args):
    if args.emotion:
        return os.path.join(
            args.data_dir_path,
            args.dataset,
            "data_" + args.dataset + "_" + args.emotion + ".pkl",
        )
    return os.path.join(args.data_dir_path, args.dataset, "data_" + args.dataset + ".pkl")


def resolve_checkpoint_path(args):
    if args.checkpoint:
        return args.checkpoint
    if args.dataset == "mosei":
        if args.emotion is None:
            raise ValueError("For mosei, please provide --emotion or --checkpoint.")
        return os.path.join(
            "./model_checkpoints",
            "mosei_best_dev_f1_model_" + args.modalities + "_" + args.emotion + ".pt",
        )
    return os.path.join(
        "./model_checkpoints",
        args.dataset + "_best_dev_f1_model_" + args.modalities + ".pt",
    )


def main(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA is unavailable, fallback to CPU.")
        args.device = "cpu"
    data_path = resolve_data_path(args)
    ckpt_path = resolve_checkpoint_path(args)
    data = load_pkl(data_path)
    model_dict = torch.load(ckpt_path)
    stored_args = model_dict["args"]
    model = model_dict["modelN_state_dict"]
    modelF = model_dict["modelF_state_dict"]
    testset = joyful.Dataset(data["test"], modelF, False, stored_args)
    test = True
    with torch.no_grad():
        golds = []
        preds = []
        for idx in tqdm(range(len(testset)), desc="test" if test else "dev"):
            data = testset[idx]
            golds.append(data["label_tensor"])
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(stored_args.device)
            y_hat = model(data, False)
            preds.append(y_hat.detach().to("cpu"))

        if stored_args.dataset == "mosei" and stored_args.emotion == "multilabel":
            golds = torch.cat(golds, dim=0).numpy()
            preds = torch.cat(preds, dim=0).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)
        else:
            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")

        if test:
            print(metrics.classification_report(golds, preds, digits=4))

            if stored_args.dataset == "mosei" and stored_args.emotion == "multilabel":
                happy = metrics.f1_score(golds[:, 0], preds[:, 0], average="weighted")
                sad = metrics.f1_score(golds[:, 1], preds[:, 1], average="weighted")
                anger = metrics.f1_score(golds[:, 2], preds[:, 2], average="weighted")
                surprise = metrics.f1_score(
                    golds[:, 3], preds[:, 3], average="weighted"
                )
                disgust = metrics.f1_score(golds[:, 4], preds[:, 4], average="weighted")
                fear = metrics.f1_score(golds[:, 5], preds[:, 5], average="weighted")

                f1 = {
                    "happy": happy,
                    "sad": sad,
                    "anger": anger,
                    "surprise": surprise,
                    "disgust": disgust,
                    "fear": fear,
                }

            print(f"F1 Score: {f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval.py")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="iemocap_4",
        choices=["iemocap", "iemocap_4", "mosei", "meld"],
        help="Dataset name.",
    )

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, resolve from dataset/modalities/emotion.",
    )

    parser.add_argument("--device", type=str, default="cuda:0", help="Computing device.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in seqcontext.py"""
    parser.add_argument(
        "--modalities",
        type=str,
        default="atv",
        # required=True,
        choices=["a", "at", "atv", "t", "v", "av"],
        help="Modalities",
    )

    # emotion
    parser.add_argument(
        "--emotion", type=str, default=None, help="emotion class for mosei"
    )

    args = parser.parse_args()
    main(args)