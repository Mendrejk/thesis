import argparse
import os
import torch
from torch import optim
from torchvision.utils import save_image
from data_preparation import prepare_data
from models import Generator, Discriminator, AudioEnhancementGAN
from feature_extractor import build_feature_extractor
from utils import estimate_memory_usage
from callbacks import LossVisualizationCallback
import time
from tqdm import tqdm
import gc

# def print_memory_summary(step):
    # print(f"\nMemory Summary at {step}:")
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    # print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

class CheckpointCallback:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, model):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        state = {
            'epoch': epoch,
            'generator_state_dict': model.generator.state_dict(),
            'discriminator_state_dict': model.discriminator.state_dict(),
            'g_optimizer': model.g_optimizer.state_dict(),
            'd_optimizer': model.d_optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return torch.load(os.path.join(checkpoint_dir, latest_checkpoint))

def train(gan, train_loader, val_loader, num_epochs=50, log_dir='./logs', device='cuda'):
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_dir)
    start_epoch = 0

    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        gan.generator.load_state_dict(checkpoint['generator_state_dict'])
        gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        gan.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        gan.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        print(f"Resuming from epoch {start_epoch}")

    checkpoint_callback = CheckpointCallback(checkpoint_dir)
    loss_visualization_callback = LossVisualizationCallback(log_dir=log_dir)

    overall_progress = tqdm(total=num_epochs, desc="Overall Progress", position=0)
    overall_progress.update(start_epoch)

    # print_memory_summary("Before training loop")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        gan.train()

        # print_memory_summary(f"Start of epoch {epoch + 1}")

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", position=1, leave=False)
        for i, batch in enumerate(train_progress):
            loss_dict = gan.train_step(batch)
            train_progress.set_postfix(g_loss=f"{loss_dict['g_loss']:.4f}", d_loss=f"{loss_dict['d_loss']:.4f}")

            # if i % 10 == 0:  # Print memory summary every 10 batches
            #     print_memory_summary(f"During epoch {epoch + 1}, batch {i}")
        #
        # print_memory_summary(f"End of epoch {epoch + 1}")

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

        checkpoint_callback.on_epoch_end(epoch, gan)
        loss_visualization_callback.on_epoch_end(epoch, gan)

        # Save sample outputs
        sample_input = next(iter(val_loader))[0].to(device)
        sample_output = gan.generator(sample_input)
        save_image(sample_output, os.path.join(log_dir, f'sample_epoch_{epoch + 1}.png'))

        overall_progress.update(1)

        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        # print_memory_summary(f"After cleanup, end of epoch {epoch + 1}")

    overall_progress.close()
    print("Training complete!")

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # print_memory_summary("After training completion")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Enhancement GAN')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
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
    data_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader, val_loader = prepare_data(converted_dir, vinyl_crackle_dir, **data_kwargs)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # print_memory_summary("After data preparation")

    # Build the GAN
    generator = Generator(input_shape=(2, 1025, 862)).to(device)
    discriminator = Discriminator(input_shape=(2, 1025, 862)).to(device)
    feature_extractor = build_feature_extractor().to(device)

    # print_memory_summary("After model initialization")

    gan = AudioEnhancementGAN(generator, discriminator, feature_extractor, accumulation_steps=8).to(device)
    gan.compile(
        g_optimizer=optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
        d_optimizer=optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    )

    # print_memory_summary("After GAN compilation")

    # Start training
    train(gan, train_loader, val_loader, num_epochs=args.epochs, log_dir=log_dir, device=device)

    print("Training complete!")

    # print_memory_summary("Final memory summary")