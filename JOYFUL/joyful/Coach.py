import copy
import time
import os
import json

import numpy as np
from numpy.core import overrides
import torch
from tqdm import tqdm
from sklearn import metrics

import joyful

log = joyful.utils.get_logger()


class Coach:
    def __init__(self, trainset, devset, testset, model, modelF, opt1 , sched1, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.modelF = modelF
        self.opt1 = opt1
        self.scheduler = sched1
        self.args = args
        self.dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
            "meld": {"Neutral": 0, "Surprise": 1, "Fear": 2, "Sadness": 3, "Joy": 4, "Disgust": 5, "Angry": 6}
        }

        if args.dataset and args.emotion == "multilabel":
            self.dataset_label_dict["mosei"] = {
                "happiness": 0,
                "sadness": 1,
                "anger": 2,
                "surprise": 3,
                "disgust": 4,
                "fear": 5,
            }

        if args.emotion == "7class":
            self.label_to_idx = {
                "Strong Negative": 0,
                "Weak Negative": 1,
                "Negative": 2,
                "Neutral": 3,
                "Positive": 4,
                "Weak Positive": 5,
                "Strong Positive": 6,
            }
        else:
            self.label_to_idx = self.dataset_label_dict[args.dataset]

        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_state = None
        self.best_test_f1 = None
        self.best_dev_acc = None
        self.best_test_acc = None
        self.best_artifact_dir = None

    def load_ckpt(self, ckpt):
        print('')

    def train(self):
        log.debug(self.model)

        best_dev_f1, best_epoch, best_state = (
            self.best_dev_f1,
            self.best_epoch,
            self.best_state,
        )

        dev_f1s = []
        test_f1s = []
        train_losses = []
        best_test_f1 = None
        best_dev_acc = None
        best_test_acc = None

        # Train
        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            dev_f1, dev_acc, dev_loss = self.evaluate()
            self.scheduler.step(dev_loss)
            test_f1, test_acc, _ = self.evaluate(test=True)
            if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                test_f1 = np.array(list(test_f1.values())).mean()
            log.info("[Dev set] [acc {:.4f}] [f1 {:.4f}]".format(dev_acc, dev_f1))

            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                best_dev_acc = dev_acc
                best_test_acc = test_acc
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                if self.args.dataset == "mosei":
                    torch.save(
                        {"args": self.args, "state_dict": self.model},
                        "model_checkpoints/mosei_best_dev_f1_model_"
                        + self.args.modalities
                        + "_"
                        + self.args.emotion
                        + ".pt",
                    )
                else:
                    torch.save({
                        "args": self.args,
                        'modelN_state_dict': self.model,
                        'modelF_state_dict': self.modelF,
                        'lr': self.scheduler._last_lr
                    }, "model_checkpoints/"
                        + self.args.dataset
                        + "_best_dev_f1_model_"
                        + self.args.modalities
                        + ".pt")

                log.info("Save the best model.")
            log.info("[Test set] [acc {:.4f}] [f1 {:.4f}]".format(test_acc, test_f1))
            dev_f1s.append(dev_f1)
            test_f1s.append(test_f1)
            train_losses.append(train_loss)

        artifact_dir = self._save_best_run_artifacts(
            best_state,
            best_epoch,
            best_dev_f1,
            best_test_f1,
            best_dev_acc,
            best_test_acc,
        )
        self.best_test_f1 = best_test_f1
        self.best_dev_acc = best_dev_acc
        self.best_test_acc = best_test_acc
        self.best_artifact_dir = artifact_dir
        log.info(
            "[Run summary] [name {}] [seed {}] [epochs {}] [best_epoch {}] [best_dev_acc {:.4f}] [best_dev_f1 {:.4f}] [best_test_acc {:.4f}] [best_test_f1 {:.4f}]".format(
                getattr(self.args, "run_name", "single"),
                self.args.seed,
                self.args.epochs,
                best_epoch if best_epoch is not None else -1,
                best_dev_acc if best_dev_acc is not None else -1.0,
                best_dev_f1 if best_dev_f1 is not None else -1.0,
                best_test_acc if best_test_acc is not None else -1.0,
                best_test_f1 if best_test_f1 is not None else -1.0,
            )
        )
        if artifact_dir is not None:
            log.info("[Run artifacts] [dir {}]".format(artifact_dir))
        if self.args.tuning:
            self.args.experiment.log_metric("best_dev_acc", best_dev_acc, epoch=epoch)
            self.args.experiment.log_metric("best_dev_f1", best_dev_f1, epoch=epoch)
            self.args.experiment.log_metric("best_test_acc", best_test_acc, epoch=epoch)
            self.args.experiment.log_metric("best_test_f1", best_test_f1, epoch=epoch)

            return best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s


        return best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0

        self.model.train()
        self.modelF.train()

        self.trainset.shuffle()
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            self.modelF.zero_grad()
            data = self.trainset[idx]
            encoderL = data['encoder_loss']
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(self.args.device)
            nll = self.model.get_loss(data,True) + 0.05*encoderL.to(self.args.device)
            epoch_loss += nll.item()
            nll.backward()
            self.opt1.step()

        end_time = time.time()
        log.info("")
        log.info(
            "[Epoch %d] [Loss: %f] [Time: %f]"
            % (epoch, epoch_loss, end_time - start_time)
        )
        return epoch_loss

    def evaluate(self, test=False):
        dev_loss = 0
        dataset = self.testset if test else self.devset
        self.model.eval()
        self.modelF.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    if not k == "utterance_texts":
                        data[k] = v.to(self.args.device)
                y_hat = self.model(data,False)
                preds.append(y_hat.detach().to("cpu"))
                nll = self.model.get_loss(data,False)
                dev_loss += nll.item()

            if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                golds = torch.cat(golds, dim=0).numpy()
                preds = torch.cat(preds, dim=0).numpy()
                f1 = metrics.f1_score(golds, preds, average="weighted")
                acc = metrics.accuracy_score(golds, preds)
                if self.args.tuning:
                    self.args.experiment.log_metric("dev_acc", acc)
            else:
                golds = torch.cat(golds, dim=-1).numpy()
                preds = torch.cat(preds, dim=-1).numpy()
                f1 = metrics.f1_score(golds, preds, average="weighted")
                acc = metrics.accuracy_score(golds, preds)

            if test:
                print(
                    metrics.classification_report(
                        golds, preds, target_names=self.label_to_idx.keys(), digits=4
                    )
                )

                if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                    happy = metrics.f1_score(
                        golds[:, 0], preds[:, 0], average="weighted"
                    )
                    sad = metrics.f1_score(golds[:, 1], preds[:, 1], average="weighted")
                    anger = metrics.f1_score(
                        golds[:, 2], preds[:, 2], average="weighted"
                    )
                    surprise = metrics.f1_score(
                        golds[:, 3], preds[:, 3], average="weighted"
                    )
                    disgust = metrics.f1_score(
                        golds[:, 4], preds[:, 4], average="weighted"
                    )
                    fear = metrics.f1_score(
                        golds[:, 5], preds[:, 5], average="weighted"
                    )

                    f1 = {
                        "happy": happy,
                        "sad": sad,
                        "anger": anger,
                        "surprise": surprise,
                        "disgust": disgust,
                        "fear": fear,
                    }
        return f1, acc, dev_loss

    def _build_artifact_dir(self):
        output_root = getattr(self.args, "output_dir", "./run_outputs")
        run_name = getattr(self.args, "run_name", "single")
        run_folder = "{}_{}_seed{}".format(
            run_name, self.args.dataset, self.args.seed
        )
        artifact_dir = os.path.join(output_root, run_folder)
        os.makedirs(artifact_dir, exist_ok=True)
        return artifact_dir

    def _save_best_run_artifacts(
        self,
        best_state,
        best_epoch,
        best_dev_f1,
        best_test_f1,
        best_dev_acc,
        best_test_acc,
    ):
        if best_state is None:
            return None

        artifact_dir = self._build_artifact_dir()
        current_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state)

        _, dev_acc, _, dev_golds, dev_preds, dev_report, dev_cm = self._evaluate_with_details(self.devset)
        _, test_acc, _, test_golds, test_preds, test_report, test_cm = self._evaluate_with_details(self.testset)

        self.model.load_state_dict(current_state)

        summary = {
            "run_name": getattr(self.args, "run_name", "single"),
            "dataset": self.args.dataset,
            "modalities": self.args.modalities,
            "seed": self.args.seed,
            "epochs": self.args.epochs,
            "best_epoch": best_epoch,
            "best_dev_acc": float(best_dev_acc) if best_dev_acc is not None else float(dev_acc),
            "best_dev_f1": float(best_dev_f1) if best_dev_f1 is not None else None,
            "best_test_acc": float(best_test_acc) if best_test_acc is not None else float(test_acc),
            "best_test_f1": float(best_test_f1) if best_test_f1 is not None else None,
        }
        with open(os.path.join(artifact_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        with open(os.path.join(artifact_dir, "classification_report_dev.txt"), "w", encoding="utf-8") as f:
            f.write(dev_report + "\n")
        with open(os.path.join(artifact_dir, "classification_report_test.txt"), "w", encoding="utf-8") as f:
            f.write(test_report + "\n")

        if getattr(dev_cm, "ndim", 2) == 2:
            np.savetxt(os.path.join(artifact_dir, "confusion_matrix_dev.csv"), dev_cm, fmt="%d", delimiter=",")
        else:
            np.save(os.path.join(artifact_dir, "confusion_matrix_dev.npy"), dev_cm)

        if getattr(test_cm, "ndim", 2) == 2:
            np.savetxt(os.path.join(artifact_dir, "confusion_matrix_test.csv"), test_cm, fmt="%d", delimiter=",")
        else:
            np.save(os.path.join(artifact_dir, "confusion_matrix_test.npy"), test_cm)

        np.save(os.path.join(artifact_dir, "labels_dev.npy"), dev_golds)
        np.save(os.path.join(artifact_dir, "preds_dev.npy"), dev_preds)
        np.save(os.path.join(artifact_dir, "labels_test.npy"), test_golds)
        np.save(os.path.join(artifact_dir, "preds_test.npy"), test_preds)
        return artifact_dir

    def _evaluate_with_details(self, dataset):
        self.model.eval()
        self.modelF.eval()
        dev_loss = 0
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="eval details"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    if not k == "utterance_texts":
                        data[k] = v.to(self.args.device)
                y_hat = self.model(data, False)
                preds.append(y_hat.detach().to("cpu"))
                nll = self.model.get_loss(data, False)
                dev_loss += nll.item()

        if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
            golds = torch.cat(golds, dim=0).numpy()
            preds = torch.cat(preds, dim=0).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)
            report = metrics.classification_report(golds, preds, digits=4)
            cm = metrics.multilabel_confusion_matrix(golds, preds)
        else:
            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)
            report = metrics.classification_report(
                golds, preds, target_names=self.label_to_idx.keys(), digits=4
            )
            cm = metrics.confusion_matrix(golds, preds)
        return f1, acc, dev_loss, golds, preds, report, cm
