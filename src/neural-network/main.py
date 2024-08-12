import keras
import os
import json
from data_preparation import prepare_data
from models import build_generator, build_discriminator_with_sn, AudioEnhancementGAN
from feature_extractor import build_feature_extractor
from utils import estimate_memory_usage
from callbacks import LossVisualizationCallback


class CheckpointCallback(keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs=None):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.json')
        state = {
            'epoch': epoch,
            'stage': self.model.current_stage,
            'alpha': float(self.model.alpha),  # Convert to Python float for JSON serialization
            'optimizer_g': self.model.g_optimizer.get_config(),
            'optimizer_d': self.model.d_optimizer.get_config(),
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(state, f)


def load_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    with open(os.path.join(checkpoint_dir, latest_checkpoint), 'r') as f:
        return json.load(f)


def progressive_training(gan, train_dataset, val_dataset, initial_epochs=50, progressive_epochs=10, total_stages=4,
                         log_dir='./logs'):
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
        gan.g_optimizer.from_config(checkpoint['optimizer_g'])
        gan.d_optimizer.from_config(checkpoint['optimizer_d'])
        print(f"Resuming from stage {start_stage}, epoch {start_epoch}")

    for stage in range(start_stage, total_stages):
        print(f"Training stage {stage + 1}/{total_stages}")

        stage_log_dir = os.path.join(log_dir, f'stage_{stage + 1}')
        os.makedirs(stage_log_dir, exist_ok=True)

        if stage == 0:
            epochs = initial_epochs
        else:
            epochs = progressive_epochs

        loss_visualization_callback = LossVisualizationCallback(log_dir=stage_log_dir)
        checkpoint_callback = CheckpointCallback(checkpoint_dir)
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'model_{epoch:02d}.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )

        history = gan.fit(
            train_dataset,
            initial_epoch=start_epoch if stage == start_stage else 0,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=[loss_visualization_callback, checkpoint_callback, model_checkpoint]
        )

        # Save the model after each stage
        gan.generator.save(os.path.join(log_dir, f'generator_stage_{stage + 1}.h5'))
        gan.discriminator.save(os.path.join(log_dir, f'discriminator_stage_{stage + 1}.h5'))

        print(f"Stage {stage + 1} complete!")
        start_epoch = 0  # Reset start_epoch for the next stage


if __name__ == "__main__":
    keras.config.set_backend("torch")  # Ensure we're using the PyTorch backend

    converted_dir = "../data/converted/stft_segments"
    vinyl_crackle_dir = "../data/vinyl_crackle/stft_segments"
    batch_size = 128  # Adjust based on your GPU memory
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    # Estimate memory usage
    memory_usage = estimate_memory_usage(batch_size)
    print(f"Estimated memory usage for batch size {batch_size}: {memory_usage:.2f} GB")

    # Prepare data
    train_dataset, val_dataset = prepare_data(converted_dir, vinyl_crackle_dir, batch_size=batch_size)

    print(f"Number of training batches: {len(train_dataset)}")
    print(f"Number of validation batches: {len(val_dataset)}")

    # Build and compile the GAN
    generator = build_generator()
    discriminator = build_discriminator_with_sn()
    feature_extractor = build_feature_extractor()

    gan = AudioEnhancementGAN(generator, discriminator, feature_extractor)
    gan.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss_weights={
            'adversarial': 1.0,
            'content': 100.0,
            'spectral_convergence': 1.0,
            'spectral_flatness': 1.0,
            'phase_aware': 1.0,
            'multi_resolution_stft': 1.0,
            'perceptual': 0.1,
            'time_frequency': 1.0
        }
    )

    # Start progressive training
    progressive_training(gan, train_dataset, val_dataset, log_dir=log_dir)

    print("Training complete!")
