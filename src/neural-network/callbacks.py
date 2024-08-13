# callbacks.py
import keras
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.optim as optim

class LossVisualizationCallback(keras.callbacks.Callback):
    def __init__(self, log_dir='./logs'):
        super().__init__()
        self.log_dir = log_dir
        self.losses = defaultdict(list)
        self.epochs = []

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
        plt.savefig(f'{self.log_dir}/losses_epoch_{epoch}.png')
        plt.close()

    def on_train_end(self, logs=None):
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
        plt.savefig(f'{self.log_dir}/final_losses.png')
        plt.close()

        # Print final loss values
        print("Final loss values:")
        for loss_name, loss_values in self.losses.items():
            print(f"{loss_name}: {loss_values[-1]:.4f}")
