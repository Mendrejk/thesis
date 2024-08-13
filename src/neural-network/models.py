import torch
import torch.nn as nn
import torch.nn.functional as F
import losses


class Generator(nn.Module):
    def __init__(self, input_shape=(2, 1025, 862), base_filters=64, num_stages=4):
        super().__init__()
        self.num_stages = num_stages
        self.current_stage = 0
        self.input_shape = input_shape
        self.base_filters = base_filters

        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, 3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2)
            )
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )

        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(1280, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(384, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(192, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2)
            )
        ])

        self.final_conv = nn.Conv2d(64, input_shape[0], 3, padding=1)

    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for i, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x)
            encoder_outputs.append(x)
            print(f"Encoder {i} output shape: {x.shape}")

        # Bottleneck
        x = self.bottleneck(x)
        print(f"Bottleneck output shape: {x.shape}")

        # Decoder with skip connections
        for i, decoder_layer in enumerate(self.decoder):
            if i < len(self.decoder) - 1:
                encoder_output = encoder_outputs[-(i + 2)]
                print(f"Decoder {i} input shape: {x.shape}, skip connection shape: {encoder_output.shape}")
                x = F.interpolate(x, size=encoder_output.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, encoder_output], dim=1)
                print(f"After concatenation shape: {x.shape}")
            else:
                print(f"Decoder {i} input shape: {x.shape}")
            x = decoder_layer(x)
            print(f"Decoder {i} output shape: {x.shape}")

        x = self.final_conv(x)
        print(f"Final output shape: {x.shape}")
        return torch.tanh(x)

    def get_current_stage_info(self):
        current_filters = self.base_filters * (2 ** self.current_stage)
        return f"Stage {self.current_stage + 1}/{self.num_stages}, Filters: {current_filters}"



class Discriminator(nn.Module):
    def __init__(self, input_shape=(2, 1025, 862), base_filters=64, num_stages=4):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = input_shape[0]
        for i in range(num_stages):
            out_channels = base_filters * (2 ** i)
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2)
            ))
            if i > 0:
                self.layers[-1].add_module('bn', nn.BatchNorm2d(out_channels))
            in_channels = out_channels

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
        self.num_stages = generator.num_stages
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

    def val_step(self, batch):
        real_input, real_target = batch
        real_input = real_input.to(self.device)
        real_target = real_target.to(self.device)

        generated_audio = self.generator(real_input)
        fake_output = self.discriminator(generated_audio)

        g_loss, _ = losses.generator_loss(real_target, generated_audio, fake_output,
                                          self.feature_extractor(real_target),
                                          self.feature_extractor(generated_audio),
                                          self.loss_weights)

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
        real_input = real_input.to(self.device)
        real_target = real_target.to(self.device)
        batch_size = real_input.size(0)
        print(f"Input shape: {real_input.shape}")

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

    def get_current_stage_info(self):
        return self.generator.get_current_stage_info()


def build_discriminator_with_sn(input_shape=(2, 1025, 862), base_filters=64, num_stages=4):
    return Discriminator(input_shape, base_filters, num_stages)
