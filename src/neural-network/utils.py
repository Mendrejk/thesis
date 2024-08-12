import numpy as np


def estimate_memory_usage(batch_size, stft_shape=(1025, 862)):
    single_stft_memory = np.prod(stft_shape) * 8 * 2
    total_memory_per_batch = single_stft_memory * batch_size * 2 * 2
    total_memory_gb = total_memory_per_batch / (1024**3)
    return total_memory_gb
