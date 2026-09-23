from abc import ABC
import numpy as np
import torch.nn as nn
import torch, os
import mlflow
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


class BaseTrainer(ABC):
    """Shared training and evaluation logic for all trainers."""

    def __init__(self, model: nn.Module, train_loader: DataLoader, validate_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device,
                 checkpoint_dir: str, patience: int = 5, tol: float = 1e-4):
        self.device = device
        # Move the model once here, so no other method has to care about the device
        self.model = model.to(device)
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.criterion = criterion

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)  # torch.save fails if the folder is missing

        self.patience = patience  # bad epochs allowed before early stop
        self.tol = tol            # minimum val loss drop that counts as an improvement

    def _train_epoch(self, test_case:bool=False) -> list:
        """Run one pass over the training set and return the loss of each batch."""
        self.model.train()  # enable dropout and batchnorm updates

        # if run for a test case
        if test_case:
            frame ,label = next(iter(self.train_loader))
            frame, label = frame.to(self.device), label.to(self.device)
            loss = self.criterion(self.model(frame), label)

            self.optimizer.zero_grad()  # clear gradients from the previous step
            loss.backward()             # compute gradients
            self.optimizer.step()       # update parameters
            return loss.item()


        losses = []
        for frame, label in self.train_loader:
            # Data must be moved every batch, unlike the model
            frame, label = frame.to(self.device), label.to(self.device)
            loss = self.criterion(self.model(frame), label)

            self.optimizer.zero_grad()  # clear gradients from the previous step
            loss.backward()             # compute gradients
            self.optimizer.step()       # update parameters
            losses.append(loss.item())  # .item() detaches from the graph and frees memory

        return losses

    @torch.no_grad()  # no gradients needed, saves memory and time
    def evaluate(self, test_case:bool=False):
        """Evaluate on the validation set.

        Returns: mean loss, accuracy, ground truth array, predictions array.
        """
        self.model.eval()  # disable dropout, use batchnorm running stats

        # if run for a test case
        if test_case:
            frame, label = next(iter(self.validate_loader))
            frame, label = frame.to(self.device), label.to(self.device)
            logits = self.model(frame)
            loss = self.criterion(logits, label).item()
            pred = logits.armax(dim=1).cpu().numpy()

            return loss, pred, label.cpu().numpy()


        losses, gt, preds = [], [], []
        for frame, label in self.validate_loader:
            frame, label = frame.to(self.device), label.to(self.device)
            logits = self.model(frame)
            losses.append(self.criterion(logits, label).item())
            # Move to CPU numpy, since sklearn and np.concatenate cannot take CUDA tensors
            preds.append(logits.argmax(dim=1).cpu().numpy())
            gt.append(label.cpu().numpy())

        # Join per-batch arrays into one array, so metrics are computed once
        gt, preds = np.concatenate(gt), np.concatenate(preds)
        return sum(losses) / len(losses), (gt == preds).mean(), gt, preds


class NonTemporalTrainer(BaseTrainer):
    """Trainer for models that take a single frame per sample."""

    def __init__(self, model, train_loader, validate_loader, optimizer, criterion,
                 device, checkpoint_dir, model_name: str = "best",
                 patience: int = 5, tol: float = 1e-4):
        # patience and tol are passed by keyword so they cannot be mixed up with model_name
        super().__init__(model, train_loader, validate_loader, optimizer, criterion,
                         device, checkpoint_dir, patience=patience, tol=tol)
        self.model_name = model_name  # checkpoint file name without extension

    def train(self, epochs: int = 50, test_case:bool=False):
        best_loss, best_acc = float("inf"), 0.0
        bad_epochs = 0  # consecutive epochs without improvement
        ckpt = os.path.join(self.checkpoint_dir, f"{self.model_name}.pt")

        for epoch in range(epochs):
            losses = self._train_epoch(test_case=test_case)
            train_loss = sum(losses) / len(losses)
            val_loss, val_acc, _, _ = self.evaluate(test_case=test_case)

            print(f"Epoch {epoch + 1}/{epochs} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | acc {val_acc:.4f}")

            if not test_case:
                mlflow.log_metric(
                    {
                    'train_loss': train_loss,
                    'val_loss' : val_loss,
                    'val_acc' : val_acc
                    },
                    step=epoch
                )
                

            # Improvement means val loss dropped by more than tol
            if  best_loss > val_loss + self.tol:
                best_loss, best_acc, bad_epochs = val_loss, val_acc, 0
                torch.save(self.model.state_dict(), ckpt)
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    print(f"Early stop at epoch {epoch + 1}")
                    break

        # Restore the best weights, not the last epoch's
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        return best_loss, best_acc

    def report(self, labels, target_names):
        """Return the classification report and confusion matrix.

        labels: class indices, target_names: class names in the same order.
        """
        _, _, gt, preds = self.evaluate()
        # labels pins all classes, so a class missing from the s/plit does not break the report
        # zero_division=0 avoids warnings for classes that are never predicted
        class_rep = classification_report(gt, preds, labels=labels,
                                          target_names=target_names, zero_division=0)
        conf_mat = confusion_matrix(gt, preds, labels=labels)
        return class_rep, conf_mat