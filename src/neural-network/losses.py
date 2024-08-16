import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys

logging.basicConfig(level=logging.ERROR, stream=sys.stdout,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def adversarial_loss(y_true, y_pred):
    return F.binary_cross_entropy_with_logits(y_pred, y_true)

def content_loss(y_true, y_pred, loss_type='l1'):
    if loss_type == 'l1':
        return torch.mean(torch.abs(y_true - y_pred))
    elif loss_type == 'l2':
        return torch.mean(torch.square(y_true - y_pred))
    else:
        raise ValueError("Invalid loss_type. Choose 'l1' or 'l2'.")

def spectral_convergence_loss(y_true, y_pred):
    return torch.norm(torch.abs(y_true) - torch.abs(y_pred)) / torch.norm(torch.abs(y_true))

def spectral_flatness_loss(y_true, y_pred):
    true_flatness = torch.exp(torch.mean(torch.log(torch.abs(y_true) + 1e-10), dim=-1)) / (torch.mean(torch.abs(y_true), dim=-1) + 1e-10)
    pred_flatness = torch.exp(torch.mean(torch.log(torch.abs(y_pred) + 1e-10), dim=-1)) / (torch.mean(torch.abs(y_pred), dim=-1) + 1e-10)
    return torch.mean(torch.abs(true_flatness - pred_flatness))

def phase_aware_loss(y_true, y_pred):
    mag_true, phase_true = torch.abs(y_true), torch.angle(y_true)
    mag_pred, phase_pred = torch.abs(y_pred), torch.angle(y_pred)

    mag_loss = torch.mean(torch.abs(mag_true - mag_pred))
    phase_loss = torch.mean(torch.abs(torch.angle(torch.exp(1j * (phase_true - phase_pred)))))

    return mag_loss + phase_loss


def multi_resolution_stft_loss(y_true, y_pred):
    # logger.debug(f"multi_resolution_stft_loss input shapes: y_true {y_true.shape}, y_pred {y_pred.shape}")
    loss = 0
    for i in range(y_true.shape[2]):  # Iterate over frequency bins
        y_true_slice = y_true[:, :, i, :]
        y_pred_slice = y_pred[:, :, i, :]
        conv_loss = spectral_convergence_loss(y_true_slice, y_pred_slice)
        mag_loss = torch.mean(torch.abs(torch.abs(y_true_slice) - torch.abs(y_pred_slice)))
        loss += conv_loss + mag_loss

    final_loss = loss / y_true.shape[2]  # Normalize by the number of frequency bins
    # logger.debug(f"Final multi_resolution_stft_loss: {final_loss.item()}")
    return final_loss

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, y_true, y_pred):
        true_features = self.feature_extractor(y_true)
        pred_features = self.feature_extractor(y_pred)
        return torch.mean(torch.square(true_features - pred_features))


def time_frequency_loss(y_true, y_pred):
    # logger.debug(f"time_frequency_loss input shapes: y_true {y_true.shape}, y_pred {y_pred.shape}")
    time_domain_loss = torch.mean(torch.abs(y_true - y_pred))
    freq_domain_loss = torch.mean(torch.abs(torch.abs(y_true) - torch.abs(y_pred)))
    total_loss = time_domain_loss + freq_domain_loss
    # logger.debug(f"time_frequency_loss: total_loss {total_loss.item()}")
    return total_loss


def snr_loss(y_true, y_pred, noise_estimate):
    """
    Compute the Signal-to-Noise Ratio (SNR) loss.

    Args:
    y_true (torch.Tensor): The target (clean) audio signal.
    y_pred (torch.Tensor): The predicted (generated) audio signal.
    noise_estimate (torch.Tensor): An estimate of the noise in the input signal.

    Returns:
    torch.Tensor: The SNR loss.
    """
    signal_power = torch.mean(y_true ** 2, dim=-1)
    noise_power = torch.mean((y_pred - y_true) ** 2, dim=-1)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return -torch.mean(snr)  # We want to maximize SNR, so we minimize negative SNR


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
            losses[loss_name] = loss_value
        except Exception as e:
            logger.error(f"Error calculating {loss_name} loss: {str(e)}")

    if feature_extractor:
        try:
            perceptual_loss_fn = PerceptualLoss(feature_extractor)
            losses['perceptual'] = perceptual_loss_fn(y_true, y_pred)
        except Exception as e:
            logger.error(f"Error calculating perceptual loss: {str(e)}")

    total_loss = sum(weights[k] * v for k, v in losses.items() if k in weights)

    d_loss = adversarial_loss(torch.ones_like(fake_output), fake_output)

    return total_loss, losses, d_loss


def discriminator_loss(real_output, fake_output):
    real_loss = adversarial_loss(torch.ones_like(real_output), real_output)
    fake_loss = adversarial_loss(torch.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss) / 2
    return total_loss
