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
                nn.Conv2d(2, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2)
            )
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256 + 128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64 + 64, 32, 4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32 + 32, 32, 4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2)
            )
        ])

        self.final_conv = nn.Conv2d(32, input_shape[0], 3, padding=1)

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
                nn.Conv2d(2, 32, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2)
            )
        ])

        self.final_conv = nn.Conv2d(128, 1, 4, padding=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)

class AudioEnhancementGAN(nn.Module):
    def __init__(self, generator, discriminator, feature_extractor=None, accumulation_steps=4):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

        # Initialize loss_weights with default values
        self.loss_weights = {
            'adversarial': 1.0,
            'content': 100.0,
            'spectral_convergence': 1.0,
            'spectral_flatness': 1.0,
            'phase_aware': 1.0,
            'multi_resolution_stft': 1.0,
            'perceptual': 1.0,
            'time_frequency': 1.0
        }

    def compile(self, g_optimizer, d_optimizer, loss_weights=None):
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        if loss_weights:
            self.loss_weights.update(loss_weights)

    def val_step(self, batch):
        real_input, real_target = batch
        real_input = real_input.to(self.device)
        real_target = real_target.to(self.device)

        generated_audio = self.generator(real_input)
        fake_output = self.discriminator(generated_audio)

        g_loss, _ = losses.generator_loss(real_target, generated_audio, fake_output,
                                          feature_extractor=self.feature_extractor,
                                          weights=self.loss_weights)

        return g_loss.item()

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

    def train_step(self, data):
        real_input, real_target = data
        real_input = real_input.to(self.device)
        real_target = real_target.to(self.device)

        # Train the discriminator
        self.d_optimizer.zero_grad()
        generated_audio = self.generator(real_input)
        real_output = self.discriminator(real_target)
        fake_output = self.discriminator(generated_audio.detach())

        d_loss_real = losses.discriminator_loss(torch.ones_like(real_output), real_output)
        d_loss_fake = losses.discriminator_loss(torch.zeros_like(fake_output), fake_output)
        gp = self.gradient_penalty(real_target, generated_audio.detach())
        d_loss = d_loss_real + d_loss_fake + 10 * gp

        d_loss = d_loss / self.accumulation_steps
        d_loss.backward()

        if (self.current_step + 1) % self.accumulation_steps == 0:
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
            "d_loss": d_loss.item() * self.accumulation_steps,
            "g_loss": g_loss.item() * self.accumulation_steps,
            **{k: v.item() for k, v in loss_components.items()},
            "gp": gp.item()
        }

    def to(self, device):
        self.device = device
        return super().to(device)

    def get_current_stage_info(self):
        return self.generator.get_current_stage_info()


def build_discriminator_with_sn(input_shape=(2, 1025, 862), base_filters=64, num_stages=4):
    return Discriminator(input_shape, base_filters, num_stages)
