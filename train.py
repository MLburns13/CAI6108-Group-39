
"""
training pipeline for the project dataset
"""

import argparse
import model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

from data import load_preprocess_project, ProjectDataset, get_transforms, make_loader, compute_pos_weights
from model import create_model

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# early stopping from ex8
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


# model checkpointing from ex8
class ModelCheckpoint:
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_score = None

    def __call__(self, val_loss):
        if self.best_score is None or val_loss < self.best_score:
            self.best_score = val_loss
            torch.save(self.model.state_dict(), self.path)

#compute accuracy
def compute_metrics(model, data_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).int()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy().astype(int)

    match = accuracy_score(all_labels, all_preds)
    hamming_acc = 1.0 - hamming_loss(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)

    return match, hamming_acc, f1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse arguments for training the image classifier."
    )
    parser.add_argument("--root", type=str, default="aggregated",)
    parser.add_argument("--batch-size",type=int,default=128,)
    parser.add_argument( "--num-workers",type=int,default=4,)
    parser.add_argument("--epochs",type=int,default=30,)
    parser.add_argument("--patience",type=int,default=5,)
    parser.add_argument("--threshold",type=float,default=0.5,)
    parser.add_argument("--checkpoint-path",type=str,default="best_model.pth",)
    return parser.parse_args()


# training loop
def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #load data
    train_x, train_y, test_x, test_y, val_x, val_y = load_preprocess_project(
        root=args.root,
        verbose=True
    )

    train_ds = ProjectDataset(train_x, train_y, transform=get_transforms(augment=True))
    val_ds   = ProjectDataset(val_x, val_y, transform=get_transforms())

    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,)
    val_loader   = make_loader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers,)
    pos_weights = compute_pos_weights(train_y, device=str(device))
    model = create_model(num_channels=3, num_outputs=12, pos_weights=pos_weights)
    model = model.to(device)

    optimizer = model.optimizer
    loss_fn   = model.loss_func  # BCE logit loss

    # call ex 8 model checkpointing and early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    checkpoint = ModelCheckpoint(model, args.checkpoint_path)

    max_epochs = args.epochs

    # train loop ex 8
    for epoch in range(max_epochs):

        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)


        exact, hamming, f1 = compute_metrics(model, val_loader, device, threshold=args.threshold)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, "
      f"All Match: {exact:.2f}, Hamming: {hamming:.2f}, F1: {f1:.2f}")

        # early stop/checkpoint logic
        checkpoint(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load best model
    early_stopping.load_best_model(model)

    # save
    torch.save(model.state_dict(), args.checkpoint_path)
    print(f"Model saved to {args.checkpoint_path}")


if __name__ == "__main__":
    train(parse_args())