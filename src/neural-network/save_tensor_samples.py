import torch
import os


def save_raw_tensor_samples(generator, val_loader, num_samples, device, log_dir, epoch):
    generator.eval()
    with torch.no_grad():
        for i in range(num_samples):
            sample_input = next(iter(val_loader))[0].to(device)
            sample_output = generator(sample_input)

            # Create a directory for the current epoch if it doesn't exist
            epoch_dir = os.path.join(log_dir, f'epoch_{epoch + 1}_samples')
            os.makedirs(epoch_dir, exist_ok=True)

            # Save the raw tensor
            sample_path = os.path.join(epoch_dir, f'sample_{i + 1}.pt')
            torch.save(sample_output.cpu(), sample_path)
    generator.train()
