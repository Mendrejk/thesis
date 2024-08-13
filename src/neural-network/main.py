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


class CheckpointCallback:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, model):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        state = {
            'epoch': epoch,
            'stage': model.current_stage,
            'alpha': float(model.alpha),
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


def progressive_training(gan, train_loader, val_loader, initial_epochs=50, progressive_epochs=10, total_stages=4,
                         log_dir='./logs', device='cuda'):
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_dir)
    start_stage = 0
    start_epoch = 0

    if checkpoint:
        start_stage = checkpoint['stage']
        start_epoch = checkpoint['epoch'] + 1
        gan.current_stage = start_stage
        gan.alpha = checkpoint['alpha']
        gan.generator.load_state_dict(checkpoint['generator_state_dict'])
        gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        gan.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        gan.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        print(f"Resuming from stage {start_stage}, epoch {start_epoch}")

    checkpoint_callback = CheckpointCallback(checkpoint_dir)
    loss_visualization_callback = LossVisualizationCallback(log_dir=log_dir)

    for stage in range(start_stage, total_stages):
        print(f"Training stage {stage + 1}/{total_stages}")

        stage_log_dir = os.path.join(log_dir, f'stage_{stage + 1}')
        os.makedirs(stage_log_dir, exist_ok=True)

        epochs = initial_epochs if stage == 0 else progressive_epochs

        for epoch in range(start_epoch, epochs):
            gan.train()
            for batch in train_loader:
                gan.train_step(batch)

            gan.eval()
            with torch.no_grad():
                val_loss = sum(gan.val_step(batch) for batch in val_loader) / len(val_loader)

            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

            checkpoint_callback.on_epoch_end(epoch, gan)
            loss_visualization_callback.on_epoch_end(epoch, gan)

            # Save sample outputs
            sample_input = next(iter(val_loader))[0].to(device)
            sample_output = gan.generator(sample_input)
            save_image(sample_output, os.path.join(stage_log_dir, f'sample_epoch_{epoch + 1}.png'))

        # Save the model after each stage
        torch.save(gan.generator.state_dict(), os.path.join(log_dir, f'generator_stage_{stage + 1}.pth'))
        torch.save(gan.discriminator.state_dict(), os.path.join(log_dir, f'discriminator_stage_{stage + 1}.pth'))

        print(f"Stage {stage + 1} complete!")
        start_epoch = 0  # Reset start_epoch for the next stage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Enhancement GAN')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")  # will use ROCm if available
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    converted_dir = "../data/converted/stft_segments"
    vinyl_crackle_dir = "../data/vinyl_crackle/stft_segments"
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    # Estimate memory usage
    memory_usage = estimate_memory_usage(args.batch_size)
    print(f"Estimated memory usage for batch size {args.batch_size}: {memory_usage:.2f} GB")

    # Prepare data
    data_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 4,
            'pin_memory': True,
        }
        data_kwargs.update(cuda_kwargs)

    train_loader, val_loader = prepare_data(converted_dir, vinyl_crackle_dir, **data_kwargs)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Build the GAN
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    feature_extractor = build_feature_extractor().to(device)

    gan = AudioEnhancementGAN(generator, discriminator, feature_extractor, accumulation_steps=8).to(device)
    gan.g_optimizer = optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    gan.d_optimizer = optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Start progressive training
    progressive_training(gan, train_loader, val_loader, log_dir=log_dir, device=device)

    print("Training complete!")