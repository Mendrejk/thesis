import torch
import torch.nn as nn
import torch.nn.functional as F
import losses

class Generator(nn.Module):
    def __init__(self, input_shape=(2, 1025, 862)):
        super().__init__()
        self.input_shape = input_shape

        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 16, 3, stride=2, padding=1),
                nn.InstanceNorm2d(16),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.InstanceNorm2d(32),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2)
            )
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(128 + 64, 32, 4, stride=2, padding=1),
                nn.InstanceNorm2d(32),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32 + 32, 16, 4, stride=2, padding=1),
                nn.InstanceNorm2d(16),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(16 + 16, 16, 4, stride=2, padding=1),
                nn.InstanceNorm2d(16),
                nn.LeakyReLU(0.2)
            )
        ])

        self.final_conv = nn.Conv2d(16, input_shape[0], 3, padding=1)

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

class Discriminator(nn.Module):
    def __init__(self, input_shape=(2, 1025, 862)):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(2, 16, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(16, 32, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2)
            )
        ])

        self.final_conv = nn.utils.spectral_norm(nn.Conv2d(64, 1, 4, padding=1))

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

        # Initialize loss_weights with default values
        self.loss_weights = {
            'adversarial': 1.0,
            'content': 10.0,
            'spectral_convergence': 0.1,
            'spectral_flatness': 0.1,
            'phase_aware': 0.1,
            'multi_resolution_stft': 1.0,
            'perceptual': 0.1,
            'time_frequency': 1.0
        }

        # New: Add discriminator update frequency
        self.d_update_frequency = 5  # Update discriminator every 5 steps

    def compile(self, g_optimizer, d_optimizer, loss_weights=None):
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        if loss_weights:
            self.loss_weights.update(loss_weights)

    def add_noise_to_input(self, x, noise_factor=0.05):
        noise = torch.randn_like(x) * noise_factor
        return x + noise

    def train_step(self, data):
        real_input, real_target = data
        real_input = real_input.to(self.device)
        real_target = real_target.to(self.device)

        # Train the discriminator (less frequently)
        if self.current_step % self.d_update_frequency == 0:
            self.d_optimizer.zero_grad()
            generated_audio = self.generator(real_input)

            # Add noise to discriminator inputs
            real_target_noisy = self.add_noise_to_input(real_target)
            generated_audio_noisy = self.add_noise_to_input(generated_audio.detach())

            real_output = self.discriminator(real_target_noisy)
            fake_output = self.discriminator(generated_audio_noisy)

            # Label smoothing
            real_labels = torch.ones_like(real_output).to(self.device) * 0.9  # Smooth positive labels
            fake_labels = torch.zeros_like(fake_output).to(self.device) * 0.1  # Smooth negative labels

            d_loss_real = losses.discriminator_loss(real_labels, real_output)
            d_loss_fake = losses.discriminator_loss(fake_labels, fake_output)
            gp = self.gradient_penalty(real_target, generated_audio.detach())
            d_loss = d_loss_real + d_loss_fake + 10 * gp

            d_loss = d_loss / self.accumulation_steps
            d_loss.backward()

            if (self.current_step + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.d_optimizer.step()
                self.d_optimizer.zero_grad()

        # Train the generator
        self.g_optimizer.zero_grad()
        generated_audio = self.generator(real_input)
        fake_output = self.discriminator(generated_audio)

        g_loss, loss_components = losses.generator_loss(
            real_target,
            generated_audio,
            fake_output,
            feature_extractor=self.feature_extractor,
            weights=self.loss_weights
        )

        g_loss = g_loss / self.accumulation_steps
        g_loss.backward()

        if (self.current_step + 1) % self.accumulation_steps == 0:
            self.g_optimizer.step()
            self.g_optimizer.zero_grad()

        self.current_step += 1

        return {
            "d_loss": d_loss.item() * self.accumulation_steps if self.current_step % self.d_update_frequency == 0 else 0,
            "g_loss": g_loss.item() * self.accumulation_steps,
            **{k: v.item() for k, v in loss_components.items()},
            "gp": gp.item() if self.current_step % self.d_update_frequency == 0 else 0
        }

    def to(self, device):
        self.device = device
        return super().to(device)

def build_discriminator_with_sn(input_shape=(2, 1025, 862)):
    return Discriminator(input_shape)