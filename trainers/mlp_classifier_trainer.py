import os
import torch
import wandb
import numpy as np
from torch import nn
from data_provider.datasets import EMGClassifierDataset
from torch.utils.data import DataLoader, Subset
from utils.registry import registry
from classifiers.ssl_cnn import EMGFeatureCNN
from classifiers.pure_cnn import EMGPureCNN
from classifiers.ssl_compact_cnn import EMGCompactFeatureCNN
from classifiers.mlp_classifier import MLPBottleneckClassifier
from utils.tools import ClassifierEarlyStopping
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from utils.tools import downsample_dataset


@registry.register_trainer("DownstreamClassifierTrainer")
class DownstreamClassifierTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{self.args.gpu}")
        self.original_label_counts_train = None
        self.downsampled_label_counts_train = None
        self.original_label_counts_val = None
        self.downsampled_label_counts_val = None
        self.original_label_counts_test = None
        self.downsampled_label_counts_test = None

    def _get_dataloader(self,):
        train_set = EMGClassifierDataset(self.args.train_h5_path)
        val_set = EMGClassifierDataset(self.args.val_h5_path)
        test_set = EMGClassifierDataset(self.args.test_h5_path)

        if self.args.enable_balance_labels:
            downsampled_indices, original_label_counts_train, downsampled_label_counts_train = downsample_dataset(
                train_set,
                downsample_label=0,
                target_ratio=self.args.downsample_target_ratio
            )
            train_set = Subset(train_set, downsampled_indices)

            self.original_label_counts_train = original_label_counts_train
            self.downsampled_label_counts_train = downsampled_label_counts_train

            downsampled_indices, original_label_counts_val, downsampled_label_counts_val = downsample_dataset(
                val_set,
                downsample_label=0,
                target_ratio=self.args.downsample_target_ratio,
                flag='val'
            )
            val_set = Subset(val_set, downsampled_indices)

            self.original_label_counts_val = original_label_counts_val
            self.downsampled_label_counts_val = downsampled_label_counts_val

            downsampled_indices, original_label_counts_test, downsampled_label_counts_test = downsample_dataset(
                test_set,
                downsample_label=0,
                target_ratio=self.args.downsample_target_ratio,
                flag='test'
            )
            test_set = Subset(test_set, downsampled_indices)

            self.original_label_counts_test = original_label_counts_test
            self.downsampled_label_counts_test = downsampled_label_counts_test

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.args.batch_size)
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size)

        return train_loader, val_loader, test_loader

    def _get_model(self,):
        model = registry.get_classifier_class(self.args.classifier)(self.args)

        return model

    def train(self,):
        train_loader, val_loader, test_loader = self._get_dataloader()
        model = self._get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()
        model = model.to(self.device)
        early_stopper = ClassifierEarlyStopping(patience=self.args.patience)

        for epoch in range(self.args.epochs):
            model.train()
            train_losses, train_preds, train_labels = [], [], []
            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_preds.extend(logits.argmax(dim=1).cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())

            train_acc = accuracy_score(train_labels, train_preds)
            wandb.log({"train_loss": np.mean(train_losses), "train_acc": train_acc}, step=epoch)

            # Validation
            model.eval()
            val_losses, val_preds, val_labels = [], [], []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_losses.append(loss.item())
                    val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    val_labels.extend(batch_y.cpu().numpy())
            val_acc = accuracy_score(val_labels, val_preds)
            wandb.log({"val_loss": np.mean(val_losses), "val_acc": val_acc}, step=epoch)
            print(f"Epoch {epoch+1} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f}")

            if early_stopper.step(model, val_acc):
                print("Early stopping triggered.")
                break

        # Save best model
        print(f"Saving best model with ValAcc: {early_stopper.best_acc:.4f}")
        ckpt_dir = os.path.join(self.args.ckpt_save_dir, self.args.exp_name)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_save_path = os.path.join(ckpt_dir, self.args.ckpt_filename)
        torch.save(early_stopper.best_model, ckpt_save_path)

        # Testing
        model.load_state_dict(torch.load(ckpt_save_path))
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = model(batch_x)
                test_preds.extend(logits.argmax(dim=1).cpu().numpy())
                test_labels.extend(batch_y.cpu().numpy())
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='macro')
        test_cm = confusion_matrix(test_labels, test_preds)
        wandb.log({
            "test_acc": test_acc,
            "test_macro_f1": test_f1,
            "test_confusion_matrix": wandb.Table(
                columns=[f"Pred_{i}" for i in range(test_cm.shape[1])],
                data=test_cm.tolist()
            )
        })
        print(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"Final Test Macro F1: {test_f1:.4f}")
        print("Confusion Matrix:\n", test_cm)

        # For better confusion matrix visualization edit on wandb (Vega Confusion Matrix)
        cm_data = []
        class_names=[str(i) for i in sorted(set(test_labels))]
        for i in range(test_cm.shape[0]):
            for j in range(test_cm.shape[1]):
                cm_data.append([class_names[j], class_names[i], test_cm[i][j]])  # [Predicted, Actual, Count]

        cm_table = wandb.Table(columns=["Predicted", "Actual", "Count"], data=cm_data)
        wandb.log({
            "vega_confusion_matrix": cm_table,  # 用于 Vega 图表自定义映射
            "confusion_matrix_plot": wandb.plot.confusion_matrix(
                y_true=test_labels,
                preds=test_preds,
                class_names=class_names
            )
        })
        

        if self.args.enable_balance_labels:
            original_table_train = wandb.Table(data=[[str(k), v] for k, v in sorted(self.original_label_counts_train.items())], columns=["label", "count"])
            original_bar_train = wandb.plot.bar(original_table_train, "label", "count", title="Original Label Distribution of Training set")
            downsampled_table_train = wandb.Table(data=[[str(k), v] for k, v in sorted(self.downsampled_label_counts_train.items())], columns=["label", "count"])
            downsampled_bar_train = wandb.plot.bar(downsampled_table_train, "label", "count", title="Downsampled Label Distribution of Training set")

            original_table_val = wandb.Table(data=[[str(k), v] for k, v in sorted(self.original_label_counts_val.items())], columns=["label", "count"])
            original_bar_val = wandb.plot.bar(original_table_val, "label", "count", title="Original Label Distribution of Val set")
            downsampled_table_val = wandb.Table(data=[[str(k), v] for k, v in sorted(self.downsampled_label_counts_val.items())], columns=["label", "count"])
            downsampled_bar_val = wandb.plot.bar(downsampled_table_val, "label", "count", title="Downsampled Label Distribution of Val set")

            original_table_test = wandb.Table(data=[[str(k), v] for k, v in sorted(self.original_label_counts_test.items())], columns=["label", "count"])
            original_bar_test = wandb.plot.bar(original_table_test, "label", "count", title="Original Label Distribution of Test set")
            downsampled_table_test = wandb.Table(data=[[str(k), v] for k, v in sorted(self.downsampled_label_counts_test.items())], columns=["label", "count"])
            downsampled_bar_test = wandb.plot.bar(downsampled_table_test, "label", "count", title="Downsampled Label Distribution of Test set")

            wandb.log({
                "original_label_distribution_train": original_bar_train,
                "downsampled_label_distribution_train": downsampled_bar_train,
                "original_label_distribution_val": original_bar_val,
                "downsampled_label_distribution_val": downsampled_bar_val,
                "original_label_distribution_test": original_bar_test,
                "downsampled_label_distribution_test": downsampled_bar_test,
            })
