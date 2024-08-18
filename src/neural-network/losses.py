import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    if not check_tensor(y_true, "spectral_convergence_loss y_true") or not check_tensor(y_pred, "spectral_convergence_loss y_pred"):
        return torch.tensor(float('nan'))
    return torch.norm(torch.abs(y_true) - torch.abs(y_pred)) / (torch.norm(torch.abs(y_true)) + 1e-7)

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
    if not check_tensor(y_true, "multi_resolution_stft_loss y_true") or not check_tensor(y_pred, "multi_resolution_stft_loss y_pred"):
        return torch.tensor(float('nan'))
    loss = 0
    for i in range(y_true.shape[2]):  # Iterate over frequency bins
        y_true_slice = y_true[:, :, i, :]
        y_pred_slice = y_pred[:, :, i, :]
        conv_loss = spectral_convergence_loss(y_true_slice, y_pred_slice)
        mag_loss = torch.mean(torch.abs(torch.abs(y_true_slice) - torch.abs(y_pred_slice)))
        loss += conv_loss + mag_loss

    final_loss = loss / y_true.shape[2]  # Normalize by the number of frequency bins
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


def snr_loss(y_true, y_pred, noise_estimate):
    # Input range check
    assert torch.all(y_true >= -1) and torch.all(y_true <= 1), "y_true out of range"
    assert torch.all(y_pred >= -1) and torch.all(y_pred <= 1), "y_pred out of range"

    # Add small epsilon to avoid division by zero
    signal_power = torch.mean(y_true ** 2, dim=-1) + 1e-6
    noise_power = torch.mean((y_pred - y_true) ** 2, dim=-1) + 1e-6

    # Clip SNR values to prevent extreme values
    snr = torch.clamp(10 * torch.log10(signal_power / (noise_power + 1e-6)), min=-100, max=100)

    # Debugging information
    logger.debug(f"SNR Loss - Signal Power: min={signal_power.min().item():.2e}, max={signal_power.max().item():.2e}")
    logger.debug(f"SNR Loss - Noise Power: min={noise_power.min().item():.2e}, max={noise_power.max().item():.2e}")
    logger.debug(f"SNR Loss - SNR: min={snr.min().item():.2f}, max={snr.max().item():.2f}")

    return -torch.mean(snr)


def generator_loss(y_true, y_pred, fake_output, noise_estimate, feature_extractor=None, weights=None):
    if weights is None:
        raise ValueError("Weights must be provided for generator loss calculation")

    losses = {}
    for loss_name, loss_fn in [
        ('adversarial', lambda: adversarial_loss(torch.ones_like(fake_output), fake_output)),
        ('content', lambda: content_loss(y_true, y_pred)),
        ('spectral_convergence', lambda: spectral_convergence_loss(y_true, y_pred)),
        ('spectral_flatness', lambda: spectral_flatness_loss(y_true, y_pred)),
        ('phase_aware', lambda: phase_aware_loss(y_true, y_pred)),
        ('multi_resolution_stft', lambda: multi_resolution_stft_loss(y_true, y_pred)),
        ('time_frequency', lambda: time_frequency_loss(y_true, y_pred)),
        ('snr', lambda: snr_loss(y_true, y_pred, noise_estimate))
    ]:
        try:
            loss_value = loss_fn()
            if torch.isnan(loss_value):
                logger.error(f"NaN detected in {loss_name} loss")
            elif torch.isinf(loss_value):
                logger.error(f"Inf detected in {loss_name} loss")
            else:
                losses[loss_name] = loss_value
        except Exception as e:
            logger.error(f"Error calculating {loss_name} loss: {str(e)}")

    if feature_extractor:
        try:
            perceptual_loss_fn = PerceptualLoss(feature_extractor)
            perceptual_loss = perceptual_loss_fn(y_true, y_pred)
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
