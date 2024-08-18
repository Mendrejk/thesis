import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import torch
import librosa
import librosa.display
import csv
from scipy import signal
from typing_extensions import override


class LossVisualizationCallback:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = defaultdict(list)
        self.loss_components = defaultdict(list)
        self.csv_file = os.path.join(log_dir, 'loss_log.csv')
        self.iteration_csv_file = os.path.join(log_dir, 'iteration_loss_log.csv')
        self.epochs = []
        self.iterations = []
        self.iteration_counter = 0
        self.initialize_csv()
        self.initialize_iteration_csv()

    def initialize_csv(self):
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['epoch', 'g_loss', 'd_loss', 'val_loss']
            writer.writerow(header)

    def initialize_iteration_csv(self):
        with open(self.iteration_csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['iteration', 'g_loss', 'd_loss']
            writer.writerow(header)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        if logs is not None:
            for k, v in logs.items():
                if k.startswith('loss_component_'):
                    component_name = k[15:]  # Remove 'loss_component_' prefix
                    self.loss_components[component_name].append(v)
                else:
                    self.losses[k].append(v)

            # Log to CSV
            self.log_to_csv(epoch, logs)

        # Only plot if we have data
        if self.epochs:
            self.plot_losses(epoch)
            self.plot_loss_components(epoch)

    # def on_train_step_end(self, state, unit):
    #     self.iteration_counter += 1
    #     g_loss = state.g_loss.item() if isinstance(state.g_loss, torch.Tensor) else state.g_loss
    #     d_loss = state.d_loss.item() if isinstance(state.d_loss, torch.Tensor) else state.d_loss
    #
    #     self.iterations.append(self.iteration_counter)
    #     self.losses['g_loss'].append(g_loss)
    #     self.losses['d_loss'].append(d_loss)
    #
    #     # Log to iteration CSV every 50 iterations
    #     if self.iteration_counter % 50 == 0:
    #         self.log_to_iteration_csv(g_loss, d_loss)

    def log_to_csv(self, epoch, logs):
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            row = [epoch, logs.get('g_loss', ''), logs.get('d_loss', ''), logs.get('val_loss', '')]

            # Add loss components
            for component, values in self.loss_components.items():
                if len(values) > 0:
                    row.append(values[-1])
                else:
                    row.append('')

            writer.writerow(row)

    def log_to_iteration_csv(self, g_loss, d_loss):
        with open(self.iteration_csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            row = [self.iteration_counter, g_loss, d_loss]
            writer.writerow(row)

    def plot_losses(self, epoch):
        if not self.epochs:  # If there's no data to plot, return early
            return

        plt.figure(figsize=(15, 10))
        plt.style.use('ggplot')

        main_losses = ['g_loss', 'd_loss', 'val_loss']
        for loss_name in main_losses:
            if loss_name in self.losses and len(self.losses[loss_name]) > 0:
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
        if not self.epochs:  # If there's no data to plot, return early
            return

        plt.figure(figsize=(15, 10))
        plt.style.use('ggplot')

        for component_name, component_values in self.loss_components.items():
            if len(component_values) > 0:
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
            if loss_values:
                print(f"{loss_name}: {loss_values[-1]:.4f}")

    def visualize_stft_comparison(self, original_stft, damaged_stft, generated_stft, epoch, sr=22050, hop_length=512):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

        # Plot original (undamaged) STFT
        img1 = librosa.display.specshow(librosa.amplitude_to_db(original_stft, ref=np.max),
                                        sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax1)
        ax1.set_title('Original (Undamaged) STFT', fontsize=16)
        fig.colorbar(img1, ax=ax1, format='%+2.0f dB')

        # Plot damaged STFT
        img2 = librosa.display.specshow(librosa.amplitude_to_db(damaged_stft, ref=np.max),
                                        sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax2)
        ax2.set_title('Damaged STFT', fontsize=16)
        fig.colorbar(img2, ax=ax2, format='%+2.0f dB')

        # Plot generated (restored) STFT
        img3 = librosa.display.specshow(librosa.amplitude_to_db(generated_stft, ref=np.max),
                                        sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax3)
        ax3.set_title('Generated (Restored) STFT', fontsize=16)
        fig.colorbar(img3, ax=ax3, format='%+2.0f dB')

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