import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import torch
import librosa
import librosa.display


class LossVisualizationCallback:
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        self.losses = defaultdict(list)
        self.epochs = []
        os.makedirs(log_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.epochs.append(epoch)
        for k, v in logs.items():
            self.losses[k].append(v)

        # Plot losses
        self.plot_losses(epoch)

    def plot_losses(self, epoch):
        plt.figure(figsize=(15, 10))

        # Use a simple, widely available style
        plt.style.use('ggplot')

        for loss_name, loss_values in self.losses.items():
            plt.plot(self.epochs, loss_values, label=loss_name, linewidth=2)

        plt.title('Training Losses', fontsize=20, fontweight='bold')
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'losses_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def on_train_end(self):
        # Plot final losses
        self.plot_losses(max(self.epochs))

        # Print final loss values
        print("Final loss values:")
        for loss_name, loss_values in self.losses.items():
            print(f"{loss_name}: {loss_values[-1]:.4f}")

    def visualize_stft_comparison(self, original_stft, generated_stft, epoch, sr=22050, hop_length=512):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot original STFT
        img1 = librosa.display.specshow(librosa.amplitude_to_db(np.abs(original_stft), ref=np.max),
                                        sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax1)
        ax1.set_title('Original STFT', fontsize=16)
        fig.colorbar(img1, ax=ax1, format='%+2.0f dB')

        # Plot generated STFT
        img2 = librosa.display.specshow(librosa.amplitude_to_db(np.abs(generated_stft), ref=np.max),
                                        sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax2)
        ax2.set_title('Generated STFT', fontsize=16)
        fig.colorbar(img2, ax=ax2, format='%+2.0f dB')

        plt.suptitle(f'STFT Comparison - Epoch {epoch}', fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'stft_comparison_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
        plt.close()

class EarlyStoppingCallback:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, epoch, val_loss, model):
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

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class CheckpointCallback:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def __call__(self, epoch, model):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        state = {
            'epoch': epoch,
            'generator_state_dict': model.generator.state_dict(),
            'discriminator_state_dict': model.discriminator.state_dict(),
            'g_optimizer': model.g_optimizer.state_dict(),
            'd_optimizer': model.d_optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)