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
import time
from tqdm import tqdm
import gc


def get_checkpoint_dirs(log_dir):
    checkpoint_dirs = [d for d in os.listdir(log_dir) if
                       os.path.isdir(os.path.join(log_dir, d)) and d.startswith('checkpoints_')]
    return sorted(checkpoint_dirs, reverse=True)


def get_checkpoints(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    return sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))


def remove_all_checkpoints(log_dir):
    checkpoint_dirs = get_checkpoint_dirs(log_dir)
    if not checkpoint_dirs:
        print("No checkpoints found.")
        return

    print("Warning: This will remove all existing checkpoints.")
    confirm = input("Are you sure you want to proceed? (y/N): ").lower()
    if confirm != 'y':
        print("Checkpoint removal cancelled.")
        return

    confirm_again = input("This action cannot be undone. Type 'DELETE' to confirm: ")
    if confirm_again != 'DELETE':
        print("Checkpoint removal cancelled.")
        return

    for dir_name in checkpoint_dirs:
        dir_path = os.path.join(log_dir, dir_name)
        shutil.rmtree(dir_path)
        print(f"Removed checkpoint directory: {dir_path}")

    print("All checkpoints have been removed.")


def select_checkpoint(log_dir):
    checkpoint_dirs = get_checkpoint_dirs(log_dir)

    if not checkpoint_dirs:
        print("No existing checkpoints found. Starting from scratch.")
        return None

    print("Available options:")
    print("0. Start from scratch")
    print("1. Remove all checkpoints")
    for i, d in enumerate(checkpoint_dirs, 2):
        print(f"{i}. {d}")

    while True:
        try:
            choice = int(input("Enter your choice (0 to start from scratch, 1 to remove all checkpoints): "))
            if 0 <= choice <= len(checkpoint_dirs) + 1:
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    if choice == 0:
        return None
    elif choice == 1:
        remove_all_checkpoints(log_dir)
        return select_checkpoint(log_dir)  # Recursively call to select after removal

    selected_dir = os.path.join(log_dir, checkpoint_dirs[choice - 2])
    checkpoints = get_checkpoints(selected_dir)

    print("\nAvailable checkpoints:")
    for i, c in enumerate(checkpoints, 1):
        print(f"{i}. {c}")

    while True:
        try:
            checkpoint_choice = int(input("Enter the number of the checkpoint to use: "))
            if 1 <= checkpoint_choice <= len(checkpoints):
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    selected_checkpoint = os.path.join(selected_dir, checkpoints[checkpoint_choice - 1])

    return torch.load(selected_checkpoint)


def train(gan, train_loader, val_loader, num_epochs=50, log_dir='./logs', device='cuda'):
    checkpoint = select_checkpoint(log_dir)
    start_epoch = 0

    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        gan.generator.load_state_dict(checkpoint['generator_state_dict'])
        gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        gan.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        gan.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        print(f"Resuming from epoch {start_epoch}")
        # Use the same checkpoint directory
        checkpoint_dir = os.path.dirname(checkpoint['checkpoint_path'])
    else:
        print("Starting training from scratch")
        # Create a new checkpoint directory
        current_date = datetime.now().strftime("%Y%m%d")
        checkpoint_dir = os.path.join(log_dir, f'checkpoints_{current_date}')
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(checkpoint_dir)
    loss_visualization_callback = LossVisualizationCallback(log_dir=log_dir)
    early_stopping_callback = EarlyStoppingCallback(patience=5, verbose=True, delta=0.01,
                                                    path=os.path.join(log_dir, 'best_model.pt'))

    overall_progress = tqdm(total=num_epochs, desc="Overall Progress", position=0)
    overall_progress.update(start_epoch)

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        gan.train()

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", position=1, leave=False)
        epoch_losses = defaultdict(float)
        for i, batch in enumerate(train_progress):
            loss_dict = gan.train_step(batch)
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            train_progress.set_postfix(g_loss=f"{loss_dict['g_loss']:.4f}", d_loss=f"{loss_dict['d_loss']:.4f}")

        # Average the losses
        avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}

        gan.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += gan.val_step(batch)
        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epochs_left = num_epochs - (epoch + 1)
        eta = epoch_duration * epochs_left / 3600  # ETA in hours

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, ETA: {eta:.2f} hours")

        # Call callbacks
        checkpoint_callback(epoch, gan)
        loss_visualization_callback.on_epoch_end(epoch, {**avg_losses, 'val_loss': val_loss})
        if early_stopping_callback(epoch, val_loss, gan):
            print("Early stopping triggered")
            break

        # Save sample outputs
        sample_input = next(iter(val_loader))[0].to(device)
        sample_output = gan.generator(sample_input)
        save_image(sample_output, os.path.join(log_dir, f'sample_epoch_{epoch + 1}.png'))
        save_raw_tensor_samples(gan.generator, val_loader, num_samples=5, device=device, log_dir=log_dir, epoch=epoch)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Enhancement GAN')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, default=18, metavar='N',
                        help='input batch size for training (default: 18)')
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

    # Estimate memory usage
    memory_usage = estimate_memory_usage(args.batch_size)
    print(f"Estimated memory usage for batch size {args.batch_size}: {memory_usage:.2f} GB")

    # Prepare data
    data_kwargs = {'batch_size': args.batch_size, 'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    train_loader, val_loader = prepare_data(converted_dir, vinyl_crackle_dir, **data_kwargs)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Build the GAN
    generator = Generator(input_shape=(2, 1025, 862)).to(device)
    discriminator = Discriminator(input_shape=(2, 1025, 862)).to(device)
    feature_extractor = build_feature_extractor().to(device)

    gan = AudioEnhancementGAN(generator, discriminator, feature_extractor, accumulation_steps=4).to(device)

    # Adjust learning rates
    g_lr = 0.0002  # Slightly higher learning rate for generator
    d_lr = 0.0001  # Lower learning rate for discriminator

    gan.compile(
        g_optimizer=optim.Adam(gan.generator.parameters(), lr=g_lr, betas=(0.5, 0.999)),
        d_optimizer=optim.Adam(gan.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
    )

    # Start training
    train(gan, train_loader, val_loader, num_epochs=args.epochs, log_dir=log_dir, device=device)

    print("Training complete!")