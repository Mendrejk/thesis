import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
import numpy as np

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("debug.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

def check_tensor(tensor, tensor_name):
    if torch.isnan(tensor).any():
        logger.error(f"NaN detected in {tensor_name}")
        return False
    if torch.isinf(tensor).any():
        logger.error(f"Inf detected in {tensor_name}")
        return False
    return True

def adversarial_loss(y_true, y_pred):
    if not check_tensor(y_true, "adversarial_loss y_true") or not check_tensor(y_pred, "adversarial_loss y_pred"):
        return torch.tensor(float('nan'))
    return F.binary_cross_entropy_with_logits(y_pred, y_true)

def content_loss(y_true, y_pred, loss_type='l1'):
    if not check_tensor(y_true, "content_loss y_true") or not check_tensor(y_pred, "content_loss y_pred"):
        return torch.tensor(float('nan'))
    if loss_type == 'l1':
        return torch.mean(torch.abs(y_true - y_pred))
    elif loss_type == 'l2':
        return torch.mean(torch.square(y_true - y_pred))
    else:
        raise ValueError("Invalid loss_type. Choose 'l1' or 'l2'.")

def spectral_convergence_loss(y_true, y_pred):
    logger.debug(f"spectral_convergence_loss input shapes: y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    if not check_tensor(y_true, "spectral_convergence_loss y_true") or not check_tensor(y_pred, "spectral_convergence_loss y_pred"):
        return torch.tensor(float('nan'))

    try:
        numerator = torch.norm(y_true - y_pred)
        denominator = torch.norm(y_true) + 1e-7
        loss = numerator / denominator
        logger.debug(f"Spectral convergence loss: {loss.item()}")
        return loss
    except Exception as e:
        logger.error(f"Error in spectral_convergence_loss: {str(e)}")
        raise


def spectral_flatness_loss(y_true, y_pred):
    if not check_tensor(y_true, "spectral_flatness_loss y_true") or not check_tensor(y_pred, "spectral_flatness_loss y_pred"):
        return torch.tensor(float('nan'))
    true_flatness = torch.exp(torch.mean(torch.log(torch.abs(y_true) + 1e-7), dim=-1)) / (torch.mean(torch.abs(y_true), dim=-1) + 1e-7)
    pred_flatness = torch.exp(torch.mean(torch.log(torch.abs(y_pred) + 1e-7), dim=-1)) / (torch.mean(torch.abs(y_pred), dim=-1) + 1e-7)
    return torch.mean(torch.abs(true_flatness - pred_flatness))

def phase_aware_loss(y_true, y_pred):
    if not check_tensor(y_true, "phase_aware_loss y_true") or not check_tensor(y_pred, "phase_aware_loss y_pred"):
        return torch.tensor(float('nan'))
    mag_true, phase_true = torch.abs(y_true), torch.angle(y_true)
    mag_pred, phase_pred = torch.abs(y_pred), torch.angle(y_pred)

    mag_loss = torch.mean(torch.abs(mag_true - mag_pred))
    phase_loss = torch.mean(torch.abs(torch.angle(torch.exp(1j * (phase_true - phase_pred)))))

    return mag_loss + phase_loss


def multi_resolution_stft_loss(y_true, y_pred):
    logger.debug(f"multi_resolution_stft_loss input shapes: y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    if not check_tensor(y_true, "multi_resolution_stft_loss y_true") or not check_tensor(y_pred,
                                                                                         "multi_resolution_stft_loss y_pred"):
        return torch.tensor(float('nan'))

    # Ensure inputs are 3D: (batch_size, frequency_bins, time)
    if y_true.dim() == 4:
        logger.debug("y_true has 4 dimensions, using only the first channel (magnitude)")
        y_true = y_true[:, 0]  # Use only the magnitude channel
    if y_pred.dim() == 4:
        logger.debug("y_pred has 4 dimensions, using only the first channel (magnitude)")
        y_pred = y_pred[:, 0]  # Use only the magnitude channel

    logger.debug(f"After potential adjustment, shapes: y_true: {y_true.shape}, y_pred: {y_pred.shape}")

    loss = 0
    num_frequency_bins = y_true.shape[1]
    for i in range(0, num_frequency_bins, 64):  # Process in chunks of 64 frequency bins
        logger.debug(f"Processing frequency bins {i} to {i + 64}")
        y_true_slice = y_true[:, i:i + 64, :]
        y_pred_slice = y_pred[:, i:i + 64, :]
        logger.debug(f"Slice shapes: y_true_slice: {y_true_slice.shape}, y_pred_slice: {y_pred_slice.shape}")

        try:
            conv_loss = spectral_convergence_loss(y_true_slice, y_pred_slice)
            mag_loss = torch.mean(torch.abs(y_true_slice - y_pred_slice))
            loss += conv_loss + mag_loss
        except Exception as e:
            logger.error(f"Error in frequency bins {i} to {i + 64}: {str(e)}")
            raise

    final_loss = loss / (num_frequency_bins // 64 + 1)  # Normalize by the number of chunks
    logger.debug(f"Final multi-resolution STFT loss: {final_loss.item()}")
    return final_loss

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, y_true, y_pred):
        if not check_tensor(y_true, "PerceptualLoss y_true") or not check_tensor(y_pred, "PerceptualLoss y_pred"):
            return torch.tensor(float('nan'))
        true_features = self.feature_extractor(y_true)
        pred_features = self.feature_extractor(y_pred)
        return torch.mean(torch.square(true_features - pred_features))

def time_frequency_loss(y_true, y_pred):
    if not check_tensor(y_true, "time_frequency_loss y_true") or not check_tensor(y_pred, "time_frequency_loss y_pred"):
        return torch.tensor(float('nan'))
    time_domain_loss = torch.mean(torch.abs(y_true - y_pred))
    freq_domain_loss = torch.mean(torch.abs(torch.abs(y_true) - torch.abs(y_pred)))
    total_loss = time_domain_loss + freq_domain_loss
    return total_loss


def snr_loss(y_true, y_pred):
    # Separate magnitude and phase
    y_true_mag, y_true_phase = y_true[:, 0], y_true[:, 1]
    y_pred_mag, y_pred_phase = y_pred[:, 0], y_pred[:, 1]

    # Calculate SNR for magnitude
    signal_power_mag = torch.mean(y_true_mag ** 2, dim=(-2, -1)) + 1e-8
    noise_power_mag = torch.mean((y_pred_mag - y_true_mag) ** 2, dim=(-2, -1)) + 1e-8
    snr_mag = 10 * torch.log10(signal_power_mag / noise_power_mag)

    # Calculate circular distance for phase
    phase_diff = torch.abs(y_true_phase - y_pred_phase)
    phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)
    phase_mse = torch.mean(phase_diff ** 2, dim=(-2, -1))

    # Combine magnitude SNR and phase MSE
    combined_loss = -torch.mean(snr_mag) + torch.mean(phase_mse)

    logger.debug(f"SNR Loss - Magnitude SNR: min={snr_mag.min().item():.2f}, max={snr_mag.max().item():.2f}")
    logger.debug(f"SNR Loss - Phase MSE: {phase_mse.mean().item():.4f}")

    return combined_loss


def wasserstein_loss(real_output, fake_output):
    return torch.mean(fake_output) - torch.mean(real_output)


def generator_loss(y_true, y_pred, fake_output, noise_estimate, feature_extractor=None, weights=None):
    logger.debug(f"generator_loss input shapes: y_true: {y_true.shape}, y_pred: {y_pred.shape}, fake_output: {fake_output.shape}")

    if weights is None:
        raise ValueError("Weights must be provided for generator loss calculation")

    losses = {}
    for loss_name, loss_fn in [
        ('adversarial', lambda: adversarial_loss(torch.ones_like(fake_output), fake_output)),
        ('content', lambda: content_loss(y_true[:, 0], y_pred[:, 0])),  # Only for magnitude
        ('spectral_convergence', lambda: spectral_convergence_loss(y_true[:, 0], y_pred[:, 0])),  # Only for magnitude
        ('spectral_flatness', lambda: spectral_flatness_loss(y_true[:, 0], y_pred[:, 0])),  # Only for magnitude
        ('phase_aware', lambda: phase_aware_loss(y_true, y_pred)),  # This handles both magnitude and phase
        ('multi_resolution_stft', lambda: multi_resolution_stft_loss(y_true[:, 0], y_pred[:, 0])),  # Only for magnitude
        ('time_frequency', lambda: time_frequency_loss(y_true[:, 0], y_pred[:, 0])),  # Only for magnitude
        ('snr', lambda: snr_loss(y_true, y_pred))  # This now handles both magnitude and phase
    ]:
        try:
            logger.debug(f"Calculating {loss_name} loss")
            loss_value = loss_fn()
            if torch.isnan(loss_value):
                logger.error(f"NaN detected in {loss_name} loss")
            elif torch.isinf(loss_value):
                logger.error(f"Inf detected in {loss_name} loss")
            else:
                losses[loss_name] = loss_value
                logger.debug(f"{loss_name} loss: {loss_value.item()}")
        except Exception as e:
            logger.error(f"Error calculating {loss_name} loss: {str(e)}")

    if feature_extractor:
        try:
            perceptual_loss_fn = PerceptualLoss(feature_extractor)
            perceptual_loss = perceptual_loss_fn(y_true[:, 0], y_pred[:, 0])  # Only for magnitude
            if torch.isnan(perceptual_loss):
                logger.error("NaN detected in perceptual loss")
            elif torch.isinf(perceptual_loss):
                logger.error("Inf detected in perceptual loss")
            else:
                losses['perceptual'] = perceptual_loss
        except Exception as e:
            logger.error(f"Error calculating perceptual loss: {str(e)}")

    total_loss = sum(weights[k] * v for k, v in losses.items() if k in weights)

    if not check_tensor(total_loss, "generator_loss total_loss"):
        return torch.tensor(float('nan')), losses, torch.tensor(float('nan'))

    d_loss = adversarial_loss(torch.ones_like(fake_output), fake_output)

    return total_loss, losses, d_loss


def discriminator_loss(real_output, fake_output):
    if not check_tensor(real_output, "discriminator_loss real_output") or not check_tensor(fake_output, "discriminator_loss fake_output"):
        return torch.tensor(float('nan'))
    real_loss = adversarial_loss(torch.ones_like(real_output), real_output)
    fake_loss = adversarial_loss(torch.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss) / 2
    return total_loss
