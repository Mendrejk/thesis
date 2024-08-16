import argparse
import os
import shutil
from collections import defaultdict
from datetime import datetime

import torch
from torch import optim
from torchvision.utils import save_image
from data_preparation import prepare_data
from models import Generator, Discriminator, AudioEnhancementGAN
from feature_extractor import build_feature_extractor
from utils import estimate_memory_usage
from callbacks import LossVisualizationCallback, EarlyStoppingCallback, CheckpointCallback
from save_tensor_samples import save_raw_tensor_samples
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from tqdm import tqdm
import gc

import logging
import matplotlib

matplotlib.use('Agg')
logging.getLogger('matplotlib.font_manager').disabled = True


def get_checkpoint_dirs(log_dir):
    # Look for run directories instead of checkpoint directories
    run_dirs = [d for d in os.listdir(log_dir) if
                os.path.isdir(os.path.join(log_dir, d)) and d.startswith('run_')]
    return sorted(run_dirs, reverse=True)

def get_checkpoints(run_dir):
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return []
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    return sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))

def select_checkpoint(log_dir):
    run_dirs = get_checkpoint_dirs(log_dir)

    if not run_dirs:
        print("No existing runs found. Starting from scratch.")
        return None, None

    print("Available runs:")
    print("0. Start from scratch (default)")
    print("1. Remove all runs")
    for i, d in enumerate(run_dirs, 2):
        print(f"{i}. {d}")

    while True:
        choice = input("Enter your choice (press Enter for default): ")
        if choice == "":
            return None, None  # Default option: start from scratch
        try:
            choice = int(choice)
            if 0 <= choice <= len(run_dirs) + 1:
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or press Enter for default.")

    if choice == 0:
        return None, None
    elif choice == 1:
        remove_all_checkpoints(log_dir)
        return select_checkpoint(log_dir)  # Recursively call to select after removal

    selected_run_dir = os.path.join(log_dir, run_dirs[choice - 2])
    checkpoints = get_checkpoints(selected_run_dir)

    if not checkpoints:
        print(f"No checkpoints found in the selected run. Starting from scratch.")
        return None, None

    print("\nAvailable checkpoints:")
    for i, c in enumerate(checkpoints, 1):
        print(f"{i}. {c}")

    while True:
        checkpoint_choice = input("Enter the number of the checkpoint to use (press Enter for latest): ")
        if checkpoint_choice == "":
            checkpoint_choice = len(checkpoints)  # Select the latest checkpoint
        try:
            checkpoint_choice = int(checkpoint_choice)
            if 1 <= checkpoint_choice <= len(checkpoints):
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or press Enter for latest.")

    selected_checkpoint = os.path.join(selected_run_dir, 'checkpoints', checkpoints[checkpoint_choice - 1])

    return torch.load(selected_checkpoint), selected_run_dir

def remove_all_checkpoints(log_dir):
    run_dirs = get_checkpoint_dirs(log_dir)
    if not run_dirs:
        print("No runs found.")
        return

    print("Warning: This will remove all existing runs and their checkpoints.")
    confirm = input("Are you sure you want to proceed? (y/N): ").lower()
    if confirm != 'y':
        print("Removal cancelled.")
        return

    confirm_again = input("This action cannot be undone. Type 'DELETE' to confirm: ")
    if confirm_again != 'DELETE':
        print("Removal cancelled.")
        return

    for dir_name in run_dirs:
        dir_path = os.path.join(log_dir, dir_name)
        shutil.rmtree(dir_path)
        print(f"Removed run directory: {dir_path}")

    print("All runs and checkpoints have been removed.")


def train(gan, train_loader, val_loader, num_epochs=50, log_dir='./logs', device='cuda'):
    checkpoint, selected_run_dir = select_checkpoint(log_dir)
    start_epoch = 0

    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        gan.generator.load_state_dict(checkpoint['generator_state_dict'])
        gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        gan.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        gan.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        print(f"Resuming from epoch {start_epoch}")
        run_log_dir = selected_run_dir
    else:
        print("Starting training from scratch")
        # Create a unique folder for this new run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log_dir = os.path.join(log_dir, f'run_{run_id}')
        os.makedirs(run_log_dir, exist_ok=True)

    # Create a new checkpoint directory for this run
    checkpoint_dir = os.path.join(run_log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(checkpoint_dir)
    loss_visualization_callback = LossVisualizationCallback(log_dir=run_log_dir)
    early_stopping_callback = EarlyStoppingCallback(patience=5, verbose=True, delta=0.01,
                                                    path=os.path.join(run_log_dir, 'best_model.pt'))

    overall_progress = tqdm(total=num_epochs, desc="Overall Progress", position=0)
    overall_progress.update(start_epoch)

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        gan.train()

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", position=1, leave=False)
        epoch_losses = defaultdict(float)
        epoch_loss_components = defaultdict(float)
        for i, batch in enumerate(train_progress):
            loss_dict = gan.train_step(batch)
            for k, v in loss_dict.items():
                if k != 'loss_components':
                    epoch_losses[k] += v
                else:
                    for comp_k, comp_v in v.items():
                        epoch_loss_components[comp_k] += comp_v
            train_progress.set_postfix(g_loss=f"{loss_dict['g_loss']:.4f}",
                                       d_loss_from_d=f"{loss_dict['d_loss_from_d']:.4f}",
                                       d_loss_from_g=f"{loss_dict['d_loss_from_g']:.4f}")

        # Average the losses
        avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        avg_loss_components = {k: v / len(train_loader) for k, v in epoch_loss_components.items()}

        # Combine avg_losses and avg_loss_components
        combined_losses = {**avg_losses, **{f'loss_component_{k}': v for k, v in avg_loss_components.items()}}

        gan.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += gan.val_step(batch)
        val_loss /= len(val_loader)

        g_scheduler = ReduceLROnPlateau(gan.g_optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        d_scheduler = ReduceLROnPlateau(gan.d_optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        g_scheduler.step(val_loss)
        d_scheduler.step(val_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epochs_left = num_epochs - (epoch + 1)
        eta = epoch_duration * epochs_left / 3600  # ETA in hours

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, ETA: {eta:.2f} hours")

        # Call callbacks
        checkpoint_callback(epoch, gan)
        loss_visualization_callback.on_epoch_end(epoch, {**combined_losses, 'val_loss': val_loss})
        if early_stopping_callback(epoch, val_loss, gan):
            print("Early stopping triggered")
            break

        gan.reset_loss_components()

        # Save sample outputs and visualize STFT comparison
        sample_input, sample_target = next(iter(val_loader))
        sample_input = sample_input.to(device)
        sample_target = sample_target.to(device)
        sample_output = gan.generator(sample_input)

        loss_visualization_callback.visualize_stft_comparison(
            sample_target[0, 0].cpu().numpy(),
            sample_output[0, 0].detach().cpu().numpy(),
            epoch
        )

        save_raw_tensor_samples(gan.generator, val_loader, num_samples=5, device=device, log_dir=run_log_dir, epoch=epoch)

        overall_progress.update(1)

        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()

    overall_progress.close()
    loss_visualization_callback.on_train_end()
    print("Training complete!")

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()

    overall_progress.close()
    loss_visualization_callback.on_train_end()
    print("Training complete!")

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()


def get_subset_fraction():
    while True:
        choice = input("Do you want to use all data or a subset? (ALL/subset): ").lower()
        if choice == 'subset':
            while True:
                try:
                    fraction = float(input("Enter the fraction of data to use (e.g., 0.01 for 1%): "))
                    if 0 < fraction <= 1:
                        return fraction
                    else:
                        print("Please enter a value between 0 and 1.")
                except ValueError:
                    print("Invalid input. Please enter a number between 0 and 1.")
        else:
            return 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Enhancement GAN')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, default=7, metavar='N',
                        help='input batch size for training (default: 7)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Using device: {device}")

    converted_dir = "../data/converted/stft_segments"
    vinyl_crackle_dir = "../data/vinyl_crackle/stft_segments"
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    # Prompt user for subset choice
    subset_fraction = get_subset_fraction()
    print(f"Using {subset_fraction * 100}% of the dataset.")

    # Estimate memory usage
    memory_usage = estimate_memory_usage(args.batch_size)
    print(f"Estimated memory usage for batch size {args.batch_size}: {memory_usage:.2f} GB")

    # Prepare data with user-chosen subset option
    data_kwargs = {'batch_size': args.batch_size, 'num_workers': 8, 'pin_memory': True, 'subset_fraction': subset_fraction} if use_cuda else {'subset_fraction': subset_fraction}
    train_loader, val_loader = prepare_data(converted_dir, vinyl_crackle_dir, **data_kwargs)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Build the GAN
    generator = Generator(input_shape=(2, 1025, 862)).to(device)
    discriminator = Discriminator(input_shape=(2, 1025, 862)).to(device)
    feature_extractor = build_feature_extractor().to(device)

    gan = AudioEnhancementGAN(generator, discriminator, feature_extractor, accumulation_steps=4).to(device)

    # Adjust learning rates
    g_lr = 0.0001  # Reduced from 0.0002
    d_lr = 0.00005  # Reduced from 0.0001

    gan.compile(
        g_optimizer=optim.Adam(gan.generator.parameters(), lr=g_lr, betas=(0.5, 0.999)),
        d_optimizer=optim.Adam(gan.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
    )

    # Start training
    train(gan, train_loader, val_loader, num_epochs=args.epochs, log_dir=log_dir, device=device)

    print("Training complete!")