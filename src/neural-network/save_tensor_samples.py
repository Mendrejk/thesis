import torch
import os


def save_raw_tensor_samples(gan, val_loader, num_samples, device, log_dir, epoch):
    gan.generator.eval()
    with torch.no_grad():
        for i in range(num_samples):
            # Get a batch from the val_loader
            batch = next(iter(val_loader))
            input_norm, _, input_original, target_original = [t.to(device) for t in batch]

            # Generate output using normalized input
            output_norm = gan.generator(input_norm)

            # Denormalize the output using the AudioEnhancementGAN's method
            output_original = gan.denormalize_stft(output_norm, input_original)

            # Create a directory for the current epoch if it doesn't exist
            epoch_dir = os.path.join(log_dir, f'epoch_{epoch + 1}_samples')
            os.makedirs(epoch_dir, exist_ok=True)

            # Save the raw tensors
            input_path = os.path.join(epoch_dir, f'input_{i + 1}.pt')
            target_path = os.path.join(epoch_dir, f'target_{i + 1}.pt')
            output_path = os.path.join(epoch_dir, f'output_{i + 1}.pt')

            torch.save(input_original.cpu(), input_path)
            torch.save(target_original.cpu(), target_path)
            torch.save(output_original.cpu(), output_path)

    gan.generator.train()