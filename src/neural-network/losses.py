import torch
import torch.nn as nn
import torch.nn.functional as F

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

def multi_resolution_stft_loss(y_true, y_pred, fft_sizes=[2048, 1024, 512, 256, 128],
                               hop_sizes=[512, 256, 128, 64, 32]):
    loss = 0
    for fft_size, hop_size in zip(fft_sizes, hop_sizes):
        stft_true = torch.stft(y_true, n_fft=fft_size, hop_length=hop_size, return_complex=True)
        stft_pred = torch.stft(y_pred, n_fft=fft_size, hop_length=hop_size, return_complex=True)
        loss += spectral_convergence_loss(stft_true, stft_pred)
        loss += torch.mean(torch.abs(torch.abs(stft_true) - torch.abs(stft_pred)))
    return loss

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, y_true, y_pred):
        true_features = self.feature_extractor(y_true)
        pred_features = self.feature_extractor(y_pred)
        return torch.mean(torch.square(true_features - pred_features))

def time_frequency_loss(y_true, y_pred):
    time_domain_loss = torch.mean(torch.abs(y_true - y_pred))

    stft_true = torch.stft(y_true, n_fft=2048, hop_length=512, return_complex=True)
    stft_pred = torch.stft(y_pred, n_fft=2048, hop_length=512, return_complex=True)
    freq_domain_loss = torch.mean(torch.abs(torch.abs(stft_true) - torch.abs(stft_pred)))

    return time_domain_loss + freq_domain_loss

def generator_loss(y_true, y_pred, fake_output, feature_extractor=None, weights=None):
    if weights is None:
        weights = {
            'adversarial': 1.0,
            'content': 100.0,
            'spectral_convergence': 1.0,
            'spectral_flatness': 1.0,
            'phase_aware': 1.0,
            'multi_resolution_stft': 1.0,
            'perceptual': 1.0,
            'time_frequency': 1.0
        }

    losses = {
        'adversarial': adversarial_loss(torch.ones_like(fake_output), fake_output),
        'content': content_loss(y_true, y_pred),
        'spectral_convergence': spectral_convergence_loss(y_true, y_pred),
        'spectral_flatness': spectral_flatness_loss(y_true, y_pred),
        'phase_aware': phase_aware_loss(y_true, y_pred),
        'multi_resolution_stft': multi_resolution_stft_loss(y_true, y_pred),
        'time_frequency': time_frequency_loss(y_true, y_pred)
    }

    if feature_extractor:
        perceptual_loss_fn = PerceptualLoss(feature_extractor)
        losses['perceptual'] = perceptual_loss_fn(y_true, y_pred)

    total_loss = sum(weights[k] * v for k, v in losses.items() if k in weights)

    return total_loss, losses

def discriminator_loss(real_output, fake_output):
    real_loss = adversarial_loss(torch.ones_like(real_output), real_output)
    fake_loss = adversarial_loss(torch.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss) / 2
    return total_loss
