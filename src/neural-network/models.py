from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import losses
import logging
logger = logging.getLogger(__name__)

class Generator(nn.Module):
    def __init__(self, input_shape=(2, 1025, 862)):
        super().__init__()
        self.input_shape = input_shape

        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 24, 3, stride=2, padding=1),
                nn.InstanceNorm2d(24),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(24, 48, 3, stride=2, padding=1),
                nn.InstanceNorm2d(48),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(48, 96, 3, stride=2, padding=1),
                nn.InstanceNorm2d(96),
                nn.LeakyReLU(0.2)
            )
        ])

        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            ResidualBlock(96),
            ResidualBlock(96),
            ResidualBlock(96)
        )

        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(96 + 96, 48, 4, stride=2, padding=1),
                nn.InstanceNorm2d(48),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(48 + 48, 24, 4, stride=2, padding=1),
                nn.InstanceNorm2d(24),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(24 + 24, 24, 4, stride=2, padding=1),
                nn.InstanceNorm2d(24),
                nn.LeakyReLU(0.2)
            )
        ])

        self.final_conv = nn.Conv2d(24, input_shape[0], 3, padding=1)

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
            encoder_output = encoder_outputs[-(i + 1)]
            x = F.interpolate(x, size=encoder_output.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, encoder_output], dim=1)
            x = decoder_layer(x)

        # Ensure the output size matches the input size
        x = F.interpolate(x, size=self.input_shape[1:], mode='bilinear', align_corners=False)
        x = self.final_conv(x)
        return torch.tanh(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x + residual

class Discriminator(nn.Module):
    def __init__(self, input_shape=(2, 1025, 862)):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(2, 24, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(24, 48, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(48, 96, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(96, 96, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2)
            )
        ])

        self.final_conv = nn.utils.spectral_norm(nn.Conv2d(96, 1, 4, padding=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)


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

        # Initialize loss_weights with default values
        self.loss_weights = {
            'adversarial': 1.0,
            'content': 15.0,  # Increased from 10.0
            'spectral_convergence': 0.2,  # Increased from 0.1
            'spectral_flatness': 0.2,  # Increased from 0.1
            'phase_aware': 0.2,  # Increased from 0.1
            'multi_resolution_stft': 1.5,  # Increased from 1.0
            'perceptual': 0.2,  # Increased from 0.1
            'time_frequency': 1.5  # Increased from 1.0
        }

        # Add discriminator update frequency
        self.d_update_frequency = 3  # Update discriminator every n steps
        self.last_d_loss = 0  # Store the last non-zero discriminator loss

        # Add thresholds for dynamic learning rate adjustment
        self.d_loss_threshold = 8.0  # Threshold for reducing discriminator learning rate
        self.g_loss_threshold = 13.5  # Threshold for increasing generator learning rate

        # Add moving average for loss tracking
        self.d_loss_ma = None
        self.g_loss_ma = None
        self.ma_beta = 0.9  # Moving average beta

    def compile(self, g_optimizer, d_optimizer, loss_weights=None):
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        if loss_weights:
            self.loss_weights.update(loss_weights)

    def add_noise_to_input(self, x, noise_factor=0.05):
        noise = torch.randn_like(x) * noise_factor
        return x + noise

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

    def train_step(self, data):
        real_input, real_target = data
        real_input = real_input.to(self.device)
        real_target = real_target.to(self.device)

        d_loss = 0
        d_loss_from_d_step = 0

        # Train the discriminator (less frequently)
        if self.current_step % self.d_update_frequency == 0:
            self.d_optimizer.zero_grad()
            generated_audio = self.generator(real_input)

            real_target_noisy = self.add_noise_to_input(real_target)
            generated_audio_noisy = self.add_noise_to_input(generated_audio.detach())

            real_output = self.discriminator(real_target_noisy)
            fake_output = self.discriminator(generated_audio_noisy)

            real_labels = torch.ones_like(real_output).to(self.device) * 0.9
            fake_labels = torch.zeros_like(fake_output).to(self.device) * 0.1

            d_loss_real = losses.discriminator_loss(real_labels, real_output)
            d_loss_fake = losses.discriminator_loss(fake_labels, fake_output)
            gp = self.gradient_penalty(real_target, generated_audio.detach())
            d_loss = d_loss_real + d_loss_fake + 10 * gp

            d_loss = d_loss / self.accumulation_steps
            d_loss.backward()

            if (self.current_step + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.d_optimizer.step()

                self.d_loss_ma = self.update_moving_average(self.d_loss_ma, d_loss.item())

                if self.d_loss_ma < self.d_loss_threshold:
                    for param_group in self.d_optimizer.param_groups:
                        param_group['lr'] *= 0.98

            d_loss_from_d_step = d_loss.item() * self.accumulation_steps
            self.last_d_loss = d_loss_from_d_step  # Store the last non-zero loss

        # Train the generator
        self.g_optimizer.zero_grad()
        generated_audio = self.generator(real_input)
        fake_output = self.discriminator(generated_audio)

        g_loss, loss_components, d_loss_from_g = losses.generator_loss(
            real_target,
            generated_audio,
            fake_output,
            feature_extractor=self.feature_extractor,
            weights=self.loss_weights
        )

        # Store individual loss components
        for key, value in loss_components.items():
            self.loss_components[key].append(value.item())

        g_loss = g_loss / self.accumulation_steps
        g_loss.backward()

        if (self.current_step + 1) % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.g_optimizer.step()

            self.g_loss_ma = self.update_moving_average(self.g_loss_ma, g_loss.item())

            if self.g_loss_ma > self.g_loss_threshold:
                for param_group in self.g_optimizer.param_groups:
                    param_group['lr'] *= 1.02

        # Prepare log message
        g_lr = self.g_optimizer.param_groups[0]['lr']
        d_lr = self.d_optimizer.param_groups[0]['lr']

        log_parts = [
            f"Step {self.current_step}",
            f"G_loss: {g_loss.item() * self.accumulation_steps:.4f}",
            f"D_loss (last update): {self.last_d_loss:.4f}",
            f"D_loss (G step): {d_loss_from_g.item():.4f}",
            f"G_loss_MA: {self.g_loss_ma:.4f}" if self.g_loss_ma is not None else "G_loss_MA: N/A",
            f"D_loss_MA: {self.d_loss_ma:.4f}" if self.d_loss_ma is not None else "D_loss_MA: N/A",
            f"G_lr: {g_lr:.6f}",
            f"D_lr: {d_lr:.6f}"
        ]
        log_msg = ", ".join(log_parts)

        logger.info(log_msg)

        self.current_step += 1

        return {
            "g_loss": g_loss.item() * self.accumulation_steps,
            "d_loss_from_d": self.last_d_loss,
            "d_loss_from_g": d_loss_from_g.item(),
            "loss_components": {k: v[-1] for k, v in self.loss_components.items()}
        }

    def reset_loss_components(self):
        self.loss_components.clear()

    def val_step(self, batch):
        real_input, real_target = batch
        real_input = real_input.to(self.device)
        real_target = real_target.to(self.device)

        generated_audio = self.generator(real_input)
        fake_output = self.discriminator(generated_audio)

        g_loss, _, _ = losses.generator_loss(real_target, generated_audio, fake_output,
                                             feature_extractor=self.feature_extractor,
                                             weights=self.loss_weights)

        return g_loss.item()

    def to(self, device):
        self.device = device
        return super().to(device)

def build_discriminator_with_sn(input_shape=(2, 1025, 862)):
    return Discriminator(input_shape)