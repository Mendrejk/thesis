import keras
from keras import layers, Model
import keras.ops as K
import torch
import torch.nn as nn
import losses

def build_generator(input_shape=(1025, 862, 2), base_filters=64, num_stages=4):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Encoder
    encoder_outputs = []
    for i in range(num_stages):
        x = layers.Conv2D(base_filters * (2 ** i), 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        encoder_outputs.append(x)

    # Bottleneck
    for _ in range(2):
        x = layers.Conv2D(base_filters * (2 ** num_stages), 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    # Decoder with skip connections
    for i in range(num_stages - 1, -1, -1):
        x = layers.Conv2DTranspose(base_filters * (2 ** i), 3, strides=2, padding='same')(x)
        encoder_output = layers.Resizing(
            K.shape(x)[1],
            K.shape(x)[2],
            interpolation='bilinear'
        )(encoder_outputs[i])
        x = layers.Concatenate()([x, encoder_output])
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    outputs = layers.Conv2D(2, 3, padding='same', activation='tanh')(x)

    return Model(inputs, outputs, name="generator")

def build_discriminator(input_shape=(1025, 862, 2), base_filters=64, num_stages=4):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for i in range(num_stages):
        x = layers.Conv2D(base_filters * (2 ** i), 4, strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        if i > 0:
            x = layers.BatchNormalization()(x)

    x = layers.Conv2D(1, 4, padding='same')(x)

    return Model(inputs, x, name="discriminator")

class AudioEnhancementGAN(keras.Model):
    def __init__(self, generator, discriminator, feature_extractor=None):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor
        self.current_stage = 0
        self.num_stages = 4
        self.alpha = 0.0

    def compile(self, g_optimizer, d_optimizer, loss_weights=None):
        super().compile()
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

    def gradient_penalty(self, real, fake):
        epsilon = K.random.uniform(shape=[K.shape(real)[0], 1, 1, 1])
        interpolated = epsilon * real + (1 - epsilon) * fake
        with keras.utils.record_gradients():
            pred = self.discriminator(interpolated)
        grads = keras.utils.get_gradients(pred, interpolated)[0]
        grad_norm = K.sqrt(K.sum(K.square(grads), axis=[1, 2, 3]))
        gradient_penalty = K.mean(K.square(grad_norm - 1))
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
        batch_size = K.shape(real_input)[0]

        # Progressive growing
        if self.alpha > 0 and self.current_stage < self.num_stages - 1:
            low_res_real = K.image.resize(real_input, (K.shape(real_input)[1] // 2, K.shape(real_input)[2] // 2))
            low_res_real = K.image.resize(low_res_real, (K.shape(real_input)[1], K.shape(real_input)[2]))
            real_input = self.alpha * real_input + (1 - self.alpha) * low_res_real

        # Train the discriminator
        self.d_optimizer.zero_grad()
        with keras.utils.record_gradients():
            generated_audio = self.generator(real_input)
            real_output = self.discriminator(real_target)
            fake_output = self.discriminator(generated_audio)

            d_loss_real = losses.discriminator_loss(K.ones_like(real_output), real_output)
            d_loss_fake = losses.discriminator_loss(K.zeros_like(fake_output), fake_output)
            gp = self.gradient_penalty(real_target, generated_audio)
            d_loss = d_loss_real + d_loss_fake + 10 * gp

        d_loss.backward()
        self.d_optimizer.step()

        # Train the generator
        self.g_optimizer.zero_grad()
        with keras.utils.record_gradients():
            generated_audio = self.generator(real_input)
            fake_output = self.discriminator(generated_audio)

            # Convert Keras tensors to PyTorch tensors for the feature extractor
            real_target_torch = torch.from_numpy(K.convert_to_numpy(real_target))
            generated_audio_torch = torch.from_numpy(K.convert_to_numpy(generated_audio))

            # Extract features
            real_features = self.feature_extractor(real_target_torch)
            fake_features = self.feature_extractor(generated_audio_torch)

            # Convert back to Keras tensors
            real_features = K.convert_to_tensor(real_features.detach().numpy())
            fake_features = K.convert_to_tensor(fake_features.detach().numpy())

            g_loss, loss_components = losses.generator_loss(
                real_target, generated_audio, fake_output,
                real_features, fake_features, self.loss_weights
            )

        g_loss.backward()
        self.g_optimizer.step()

        # Progressive growing step
        self.progressive_step()

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            **loss_components,
            "gp": gp,
            "stage": self.current_stage,
            "alpha": self.alpha
        }

def build_discriminator_with_sn(input_shape=(1025, 862, 2), base_filters=64, num_stages=4): #sn disabled
    return build_discriminator(input_shape, base_filters, num_stages)