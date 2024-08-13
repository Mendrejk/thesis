import matplotlib.pyplot as plt
from collections import defaultdict
import os
import torch

class LossVisualizationCallback:
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        self.losses = defaultdict(list)
        self.epochs = []
        os.makedirs(log_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        for k, v in logs.items():
            self.losses[k].append(v)

        # Plot losses
        plt.figure(figsize=(15, 10))
        for loss_name, loss_values in self.losses.items():
            plt.plot(self.epochs, loss_values, label=loss_name)

        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.savefig(os.path.join(self.log_dir, f'losses_epoch_{epoch}.png'))
        plt.close()

    def on_train_end(self):
        # Plot final losses
        plt.figure(figsize=(15, 10))
        for loss_name, loss_values in self.losses.items():
            plt.plot(self.epochs, loss_values, label=loss_name)

        plt.title('Final Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Save the final plot
        plt.savefig(os.path.join(self.log_dir, 'final_losses.png'))
        plt.close()

        # Print final loss values
        print("Final loss values:")
        for loss_name, loss_values in self.losses.items():
            print(f"{loss_name}: {loss_values[-1]:.4f}")

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss