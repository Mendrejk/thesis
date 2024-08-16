import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import torch
import librosa
import librosa.display
import csv
from scipy import signal

class LossVisualizationCallback:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = defaultdict(list)
        self.loss_components = {}
        self.csv_file = os.path.join(log_dir, 'loss_log.csv')
        self.epochs = []  # Add this line
        self.initialize_csv()

    def initialize_csv(self):
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['epoch', 'g_loss', 'd_loss', 'val_loss']
            writer.writerow(header)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)  # Add this line
        if logs is not None:
            for k, v in logs.items():
                if k.startswith('loss_component_'):
                    component_name = k[15:]  # Remove 'loss_component_' prefix
                    if component_name not in self.loss_components:
                        self.loss_components[component_name] = []
                    self.loss_components[component_name].append(v)
                else:
                    self.losses[k].append(v)

            # Log to CSV
            self.log_to_csv(epoch, logs)

        self.plot_losses(epoch)
        self.plot_loss_components(epoch)

    def log_to_csv(self, epoch, logs):
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            row = [epoch, logs.get('g_loss', ''), logs.get('d_loss', ''), logs.get('val_loss', '')]

            # Add loss components
            for component in self.loss_components.keys():
                if f'loss_component_{component}' not in self.csv_file:
                    # If this is a new component, add it to the header
                    with open(self.csv_file, 'r') as read_file:
                        header = next(csv.reader(read_file))
                    header.append(f'loss_component_{component}')
                    with open(self.csv_file, 'w', newline='') as write_file:
                        csv.writer(write_file).writerow(header)

                row.append(logs.get(f'loss_component_{component}', ''))

            writer.writerow(row)

    def plot_losses(self, epoch):
        plt.figure(figsize=(15, 10))
        plt.style.use('ggplot')

        main_losses = ['g_loss', 'd_loss_from_d', 'd_loss_from_g', 'val_loss']
        for loss_name in main_losses:
            if loss_name in self.losses:
                plt.plot(self.epochs, self.losses[loss_name], label=loss_name, linewidth=2)

        plt.title('Main Training Losses', fontsize=20, fontweight='bold')
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'main_losses_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_loss_components(self, epoch):
        plt.figure(figsize=(15, 10))
        plt.style.use('ggplot')

        for component_name, component_values in self.loss_components.items():
            plt.plot(self.epochs, component_values, label=component_name, linewidth=2)

        plt.title('Individual Loss Components', fontsize=20, fontweight='bold')
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'loss_components_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def on_train_end(self):
        # Plot final losses
        self.plot_losses(max(self.epochs))
        self.plot_loss_components(max(self.epochs))

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