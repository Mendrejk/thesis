from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

import losses
import logging
logger = logging.getLogger(__name__)

# Set numba logger to WARNING level
logging.getLogger('numba').setLevel(logging.WARNING)

# Set matplotlib logger to WARNING level to reduce clutter
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class Generator(nn.Module):
    def __init__(self, input_shape=(2, 1025, 862)):
        super().__init__()
        self.input_shape = input_shape

        # Encoder
        self.encoder = nn.ModuleList([
            self.encoder_block(2, 32),
            self.encoder_block(32, 64),
            self.encoder_block(64, 128),
            self.encoder_block(128, 256)
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # Decoder
        self.decoder = nn.ModuleList([
            self.decoder_block(256 + 256, 128),
            self.decoder_block(128 + 128, 64),
            self.decoder_block(64 + 64, 32),
            self.decoder_block(32 + 32, 32)
        ])

        self.final_conv = spectral_norm(nn.Conv2d(32, input_shape[0], 3, padding=1))

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_outputs.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        for i, decoder_layer in enumerate(self.decoder):
            x = F.interpolate(x, size=encoder_outputs[-i-1].shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, encoder_outputs[-i-1]], dim=1)
            x = decoder_layer(x)

        # Ensure the output size matches the input size
        x = F.interpolate(x, size=self.input_shape[1:], mode='bilinear', align_corners=False)
        x = self.final_conv(x)
        return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, input_shape=(2, 1025, 862)):
        super().__init__()
        self.layers = nn.ModuleList([
            self.discriminator_block(2, 32),
            self.discriminator_block(32, 64),
            self.discriminator_block(64, 128),
            self.discriminator_block(128, 256),
            self.discriminator_block(256, 512)
        ])

        self.final_conv = spectral_norm(nn.Conv2d(512, 1, 4, padding=1))

    def discriminator_block(self, in_channels, out_channels):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x + residual


class AudioEnhancementGAN(nn.Module):
    def __init__(self, generator, discriminator, feature_extractor=None, accumulation_steps=1):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.loss_history = defaultdict(list)
        self.loss_components = defaultdict(list)

        self.loss_weights = {
            'adversarial': 1.0,
            'wasserstein': 1.0,  # Added Wasserstein loss weight
            'content': 10.0,  # Increased
            'spectral_convergence': 0.1,  # Decreased
            'spectral_flatness': 0.1,  # Decreased
            'phase_aware': 0.1,  # Decreased
            'multi_resolution_stft': 1.0,  # Decreased
            'perceptual': 0.1,  # Decreased
            'time_frequency': 1.0,  # Decreased
            'snr': 1.0  # Decreased
        }

        self.d_update_frequency = 1
        self.d_update_counter = 0  # Counter to keep track of iterations
        self.last_d_loss = 0

        self.d_loss_threshold = 0.5
        self.g_loss_threshold = 1.0

        self.d_loss_ma = None
        self.g_loss_ma = None
        self.ma_beta = 0.9

        self.instance_noise = 0.1
        self.instance_noise_anneal_rate = 0.99

    def compile(self, g_optimizer, d_optimizer, loss_weights=None):
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        if loss_weights:
            self.loss_weights.update(loss_weights)

    def add_instance_noise(self, x):
        return x + torch.randn_like(x) * self.instance_noise

    def gradient_penalty(self, real, fake):
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real.device)
        interpolated = epsilon * real + (1 - epsilon) * fake
        interpolated.requires_grad_(True)
        pred = self.discriminator(interpolated)
        grad = torch.autograd.grad(outputs=pred, inputs=interpolated,
                                   grad_outputs=torch.ones_like(pred),
                                   create_graph=True, retain_graph=True)[0]
        grad_norm = grad.norm(2, dim=1)
        gradient_penalty = ((grad_norm - 1) ** 2).mean()
        return gradient_penalty

    def update_moving_average(self, ma, new_value):
        if ma is None:
            return new_value
        return self.ma_beta * ma + (1 - self.ma_beta) * new_value

    def check_tensor(self, tensor, tensor_name):
        if torch.isnan(tensor).any():
            logger.error(f"NaN detected in {tensor_name}")
            return False
        if torch.isinf(tensor).any():
            logger.error(f"Inf detected in {tensor_name}")
            return False
        return True

    def denormalize_stft(self, normalized_stft, original_stft):
        # Denormalize the magnitude
        mag_norm, phase = normalized_stft[:, 0], normalized_stft[:, 1]
        mag_original = original_stft[:, 0]

        mag_min = mag_original.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        mag_max = mag_original.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        mag_denorm = (mag_norm + 1) / 2 * (mag_max - mag_min) + mag_min

        return torch.stack([mag_denorm, phase], dim=1)

    def train_step(self, data):
        real_input_norm, real_target_norm, real_input_original, real_target_original = data
        real_input_norm = real_input_norm.to(self.device)
        real_target_norm = real_target_norm.to(self.device)
        real_input_original = real_input_original.to(self.device)
        real_target_original = real_target_original.to(self.device)

        logger.debug(
            f"Train - Real input mag range: [{real_input_norm[:, 0].min().item():.4f}, {real_input_norm[:, 0].max().item():.4f}]")
        logger.debug(
            f"Train - Real input phase range: [{real_input_norm[:, 1].min().item():.4f}, {real_input_norm[:, 1].max().item():.4f}]")
        logger.debug(
            f"Train - Real target mag range: [{real_target_norm[:, 0].min().item():.4f}, {real_target_norm[:, 0].max().item():.4f}]")
        logger.debug(
            f"Train - Real target phase range: [{real_target_norm[:, 1].min().item():.4f}, {real_target_norm[:, 1].max().item():.4f}]")

        if not self.check_tensor(real_input_norm, "real_input_norm") or not self.check_tensor(real_target_norm,
                                                                                              "real_target_norm"):
            return self.return_nan_results()

        noise_estimate = real_input_original - real_target_original

        # Train the discriminator
        d_loss_from_d_step = 0
        if self.d_update_counter % self.d_update_frequency == 0:
            self.d_optimizer.zero_grad()
            generated_audio_norm = self.generator(real_input_norm)

            if not self.check_tensor(generated_audio_norm, "generated_audio_norm (discriminator step)"):
                return self.return_nan_results()

            real_target_noisy = self.add_instance_noise(real_target_norm)
            generated_audio_noisy = self.add_instance_noise(generated_audio_norm.detach())

            real_output = self.discriminator(real_target_noisy)
            fake_output = self.discriminator(generated_audio_noisy)

            d_loss_real = F.relu(1.0 - real_output).mean()
            d_loss_fake = F.relu(1.0 + fake_output).mean()
            d_loss = d_loss_real + d_loss_fake
            gp = self.gradient_penalty(real_target_norm, generated_audio_norm.detach())
            d_loss = d_loss + 10 * gp

            if not self.check_tensor(d_loss, "d_loss"):
                return self.return_nan_results()

            d_loss = d_loss / self.accumulation_steps
            d_loss.backward()

            if (self.current_step + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.d_optimizer.step()

                self.d_loss_ma = self.update_moving_average(self.d_loss_ma, d_loss.item())

                if self.d_loss_ma < self.d_loss_threshold:
                    for param_group in self.d_optimizer.param_groups:
                        param_group['lr'] *= 0.99

            d_loss_from_d_step = d_loss.item() * self.accumulation_steps
            self.last_d_loss = d_loss_from_d_step

        # Train the generator
        self.g_optimizer.zero_grad()
        generated_audio_norm = self.generator(real_input_norm)

        if not self.check_tensor(generated_audio_norm, "generated_audio_norm (generator step)"):
            return self.return_nan_results()

        fake_output = self.discriminator(generated_audio_norm)

        if not self.check_tensor(fake_output, "fake_output (generator step)"):
            return self.return_nan_results()

        # Hinge loss for generator
        g_loss_adv = -fake_output.mean()

        # Denormalize generated audio for loss calculation
        generated_audio_original = self.denormalize_stft(generated_audio_norm, real_input_original)

        g_loss, loss_components, _ = losses.generator_loss(
            real_target_original,
            generated_audio_original,
            fake_output,
            noise_estimate,
            feature_extractor=self.feature_extractor,
            weights=self.loss_weights
        )

        if not self.check_tensor(g_loss, "g_loss"):
            return self.return_nan_results()

        g_loss = g_loss + g_loss_adv

        for key, value in loss_components.items():
            if not self.check_tensor(value, f"loss_component_{key}"):
                return self.return_nan_results()
            self.loss_components[key].append(value.item())

        g_loss = g_loss / self.accumulation_steps
        g_loss.backward()

        if (self.current_step + 1) % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.g_optimizer.step()

            self.g_loss_ma = self.update_moving_average(self.g_loss_ma, g_loss.item())

            if self.g_loss_ma > self.g_loss_threshold:
                for param_group in self.g_optimizer.param_groups:
                    param_group['lr'] *= 1.01

        print(f"Step {self.current_step}: G_loss: {g_loss.item() * self.accumulation_steps:.4f}, "
              f"D_loss: {self.last_d_loss:.4f}, G_loss_adv: {g_loss_adv.item():.4f}")

        self.current_step += 1
        self.d_update_counter += 1
        self.instance_noise *= self.instance_noise_anneal_rate

        return {
            "g_loss": g_loss.item() * self.accumulation_steps,
            "d_loss_from_d": self.last_d_loss,
            "g_loss_adv": g_loss_adv.item(),
            "loss_components": {k: v[-1] for k, v in self.loss_components.items()}
        }

    def reset_loss_components(self):
        self.loss_components.clear()

    def val_step(self, batch):
        real_input_norm, real_target_norm, real_input_original, real_target_original = batch
        real_input_norm = real_input_norm.to(self.device)
        real_target_norm = real_target_norm.to(self.device)
        real_input_original = real_input_original.to(self.device)
        real_target_original = real_target_original.to(self.device)

        if not self.check_tensor(real_input_norm, "val_real_input_norm") or not self.check_tensor(real_target_norm,
                                                                                                  "val_real_target_norm"):
            return float('nan')

        generated_audio_norm = self.generator(real_input_norm)

        if not self.check_tensor(generated_audio_norm, "val_generated_audio_norm"):
            return float('nan')

        # Denormalize generated audio for loss calculation
        generated_audio_original = self.denormalize_stft(generated_audio_norm, real_input_original)

        fake_output = self.discriminator(generated_audio_norm)

        if not self.check_tensor(fake_output, "val_fake_output"):
            return float('nan')

        noise_estimate = real_input_original - real_target_original

        g_loss, _, _ = losses.generator_loss(real_target_original, generated_audio_original, fake_output,
                                             noise_estimate,
                                             feature_extractor=self.feature_extractor,
                                             weights=self.loss_weights)

        if not self.check_tensor(g_loss, "val_g_loss"):
            return float('nan')

        return g_loss.item()

    def return_nan_results(self):
        return {
            "g_loss": float('nan'),
            "d_loss_from_d": float('nan'),
            "g_loss_adv": float('nan'),
            "loss_components": {k: float('nan') for k in self.loss_components.keys()}
        }

    def to(self, device):
        self.device = device
        return super().to(device)

def build_discriminator_with_sn(input_shape=(2, 1025, 862)):
    return Discriminator(input_shape)