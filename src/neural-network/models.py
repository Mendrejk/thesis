import torch
import torch.nn as nn
import torch.nn.functional as F
import losses
from torch.utils.checkpoint import checkpoint


class Generator(nn.Module):
    def __init__(self, input_shape=(2, 1025, 862), base_filters=64, num_stages=4):
        super().__init__()
        self.num_stages = num_stages

        # Encoder
        self.encoder = nn.ModuleList()
        in_channels = input_shape[0]
        for i in range(num_stages):
            out_channels = base_filters * (2 ** i)
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            ))
            in_channels = out_channels

        # Bottleneck
        bottleneck_channels = base_filters * (2 ** num_stages)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, 3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.LeakyReLU(0.2)
        )

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_stages - 1, -1, -1):
            in_channels = base_filters * (2 ** (i + 1))
            out_channels = base_filters * (2 ** i)
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels * 2, out_channels, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            ))

        self.final_conv = nn.Conv2d(base_filters, input_shape[0], 3, padding=1)

    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for encoder_layer in self.encoder:
            x = checkpoint(encoder_layer, x, use_reentrant=False)
            encoder_outputs.append(x)

        # Bottleneck
        x = checkpoint(self.bottleneck, x, use_reentrant=False)

        # Decoder with skip connections
        for i, decoder_layer in enumerate(self.decoder):
            encoder_output = encoder_outputs[-(i + 1)]
            x = F.interpolate(x, size=encoder_output.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, encoder_output], dim=1)
            x = checkpoint(decoder_layer, x, use_reentrant=False)

        return torch.tanh(self.final_conv(x))


class Discriminator(nn.Module):
    def __init__(self, input_shape=(2, 1025, 862), base_filters=64, num_stages=4):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = input_shape[0]
        for i in range(num_stages):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, base_filters * (2 ** i), 4, stride=2, padding=1),
                nn.LeakyReLU(0.2)
            ))
            if i > 0:
                self.layers[-1].add_module('bn', nn.BatchNorm2d(base_filters * (2 ** i)))
            in_channels = base_filters * (2 ** i)

        self.final_conv = nn.Conv2d(in_channels, 1, 4, padding=1)

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
        self.current_stage = 0
        self.num_stages = 4
        self.alpha = 0.0
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def compile(self, g_optimizer, d_optimizer, loss_weights=None):
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_weights = loss_weights if loss_weights else {
            'adversarial': 1.0,
            'content': 100.0,
            'spectral_convergence': 1.0,
            'spectral_flatness': 1.0,
            'phase_aware': 1.0,
            'multi_resolution_stft': 1.0,
            'perceptual': 1.0,
            'time_frequency': 1.0
        }

    def val_step(self, data):
        real_input, real_target = data
        with torch.no_grad():
            generated_audio = self.generator(real_input)
            fake_output = self.discriminator(generated_audio)

            real_features = self.feature_extractor(real_target)
            fake_features = self.feature_extractor(generated_audio)

            g_loss, loss_components = losses.generator_loss(
                real_target, generated_audio, fake_output,
                real_features, fake_features, self.loss_weights
            )

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

    def progressive_step(self):
        self.alpha += 0.1
        if self.alpha >= 1.0:
            self.alpha = 0.0
            self.current_stage += 1
            if self.current_stage >= self.num_stages:
                self.current_stage = self.num_stages - 1

    def train_step(self, data):
        real_input, real_target = data
        real_input = real_input.to(self.device).requires_grad_(True)
        real_target = real_target.to(self.device).requires_grad_(True)
        batch_size = real_input.size(0)

        # Progressive growing
        if self.alpha > 0 and self.current_stage < self.num_stages - 1:
            low_res_real = F.interpolate(real_input, scale_factor=0.5, mode='bilinear')
            low_res_real = F.interpolate(low_res_real, size=real_input.shape[2:], mode='bilinear')
            real_input = self.alpha * real_input + (1 - self.alpha) * low_res_real

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

        # Extract features
        with torch.no_grad():
            real_features = self.feature_extractor(real_target)
        fake_features = self.feature_extractor(generated_audio)

        g_loss, loss_components = losses.generator_loss(
            real_target, generated_audio, fake_output,
            real_features, fake_features, self.loss_weights
        )

        g_loss = g_loss / self.accumulation_steps
        g_loss.backward()

        if (self.current_step + 1) % self.accumulation_steps == 0:
            self.g_optimizer.step()
            self.g_optimizer.zero_grad()

        # Progressive growing step
        self.progressive_step()

        self.current_step += 1

        return {
            "d_loss": d_loss.item() * self.accumulation_steps,
            "g_loss": g_loss.item() * self.accumulation_steps,
            **{k: v.item() for k, v in loss_components.items()},
            "gp": gp.item(),
            "stage": self.current_stage,
            "alpha": self.alpha
        }

    def to(self, device):
        self.device = device
        return super().to(device)


def build_discriminator_with_sn(input_shape=(2, 1025, 862), base_filters=64, num_stages=4):
    return Discriminator(input_shape, base_filters, num_stages)
