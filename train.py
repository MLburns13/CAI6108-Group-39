
"""
training pipeline for the project dataset
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

# Custom Imports
from data import load_preprocess_project, ProjectDataset, get_transforms, make_loader, compute_pos_weights
from model import create_model

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Deprecated from training
NUM_EPOCHS=250
BATCH_SIZE=128
LOAD_FROM_FILE=False
GRAN=False
THRESHOLD=0.6

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

#computer_acc
def compute_metrics(model, data_loader, device, threshold=0.5, granular_f1=False):
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



    f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
    precision = tp / (tp+fp+1e-8)
    recall = tp / (tp+fn+1e-8)

    if granular_f1:
        #print f1 for every label
        f1s = []
        for i in range(all_preds.shape[1]):
            tp = ((all_preds[:,i] == 1) & (all_labels[:,i] == 1)).sum().float()
            fp = ((all_preds[:,i] == 1) & (all_labels[:,i] == 0)).sum().float()
            fn = ((all_preds[:,i] == 0) & (all_labels[:,i] == 1)).sum().float()

            f1s.append(round((2 * tp / (2 * tp + fp + fn + 1e-8)).item(),2))
        
        return match, hamming_acc, f1, f1s


    return match, hamming_acc, f1, precision, recall

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

# Plot the model's performance during training (across epochs) - pulled from HW4 - slightly modified
def plot_training_perf(train_loss, val_loss, train_acc=None, val_acc=None, ax=None, fs=(5.5,2.8), plot_acc=True):
    no_ax_provided = ax == None
    if no_ax_provided:
        fig = plt.figure(figsize=fs)
        ax = plt.gca()
    
    # assume we have one measurement per epoch
    num_epochs = train_loss.shape[0]
    epochs = np.arange(0, num_epochs)

    if plot_acc:
        ax.plot(1+epochs, train_acc, 'r--', linewidth=1.5, label='Training')
        ax.plot(1+epochs, val_acc, 'b-', linewidth=1.5, label='Validation')
        
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
    else:
        ax.plot(1+epochs, train_loss, 'r--', linewidth=1.5, label='Training')
        ax.plot(1+epochs, val_loss, 'b-', linewidth=1.5, label='Validation')
        
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
    
    ax.set_xlim([1, num_epochs])

    if plot_acc:
        ylim = [0.0, 1.01]
    else:
        ylim = [0.0, 1.26]
    ax.set_ylim(ylim)

    ax.legend()
    if no_ax_provided:
        #plt.show()
        plt.tight_layout()
        plt.savefig('loss_curve.png')

def plot_lr(steps, lr, ax=None, fs=(5.5,2.8)):
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(steps, lr, label='ReduceOnLRPlateau')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.legend()

    plt.tight_layout()
    plt.savefig('lr.png')


# training loop
def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #load data
    train_x, train_y, test_x, test_y, val_x, val_y = load_preprocess_project(
        root=args.root,
        verbose=True
    )

    train_ds = ProjectDataset(train_x, train_y, transform=get_transforms(augment=True, mode='strong'))
    val_ds   = ProjectDataset(val_x, val_y, transform=get_transforms())

    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,)
    val_loader   = make_loader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers,)
    pos_weights = compute_pos_weights(train_y, device=str(device))
    model = create_model(num_channels=3, num_labels=12, drop_rate=0.4, decay=1e-3, 
                        learning_rate=0.001, pos_weights=pos_weights)

    if LOAD_FROM_FILE: # Load checkpointed file
        state_dict = torch.load('2.pth', map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    optimizer = model.optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-8, patience=5)

    loss_fn   = model.loss_func  # BCE logit loss

    # call ex 8 model checkpointing and early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    checkpoint = ModelCheckpoint(model, args.checkpoint_path)

    max_epochs = args.epochs

    train_losses = []
    val_losses = []

    steps = []
    lrs = []
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
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)


        exact, hamming, f1, prec, recall = compute_metrics(model, val_loader, device, threshold=args.threshold)
        print(f"Epoch {epoch+1} -- Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, "
        f"All Match: {exact:.2f}, Hamming: {hamming:.2f}, F1: {f1:.2f}, Precision: {prec:.2f}, Recall: {recall:.2f}")


        scheduler.step(val_loss)
        lrs.append(scheduler.get_last_lr()[0])
        steps.append(epoch)

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

    plot_lr(steps, lrs)
    plot_training_perf(np.array(train_losses), np.array(val_losses), plot_acc=False)


if __name__ == "__main__":
    train(parse_args())