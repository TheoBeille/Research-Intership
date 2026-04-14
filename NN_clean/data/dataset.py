

import numpy as np
import torch

from Algo_setuptorch import get_setup, build_algo_functions


def load_sample(seed, params, device, noise_range=(0.05, 0.2), clean_noise=0.0):
    """
    Build one sample:
        noisy image / clean image / algorithmic functions

    Args:
        seed: random seed for the sample
        params: parameter object used to build the algorithmic functions
        device: torch device
        noise_range: interval for random noise level
        clean_noise: noise level used for the clean reference

    Returns:
        noisy, clean, functions
    """
    noise_level = np.random.uniform(*noise_range)

    setup_noisy = get_setup(seed=seed, noise_level=noise_level, device=device)
    setup_clean = get_setup(seed=seed, noise_level=clean_noise, device=device)

    functions = build_algo_functions(setup_noisy, params)

    noisy = setup_noisy["noisy"].unsqueeze(1).to(device)
    clean = setup_clean["noisy"].unsqueeze(1).to(device)

    return noisy, clean, functions


def build_dataset(seeds, params, device, noise_range=(0.05, 0.2), clean_noise=0.0):
    """
    Build a full in-memory dataset as a list of samples.

    Returns:
        list of tuples (noisy, clean, functions)
    """
    data = [
        load_sample(
            seed=s,
            params=params,
            device=device,
            noise_range=noise_range,
            clean_noise=clean_noise,
        )
        for s in seeds
    ]
    return data


def split_seeds(train_seeds, test_seeds):
    """
    Small helper for readability.
    """
    return list(train_seeds), list(test_seeds)


def build_train_test_data(train_seeds, test_seeds, params, device, noise_range=(0.05, 0.2)):
    """
    Convenience function to create both train and test datasets.
    """
    train_data = build_dataset(
        seeds=train_seeds,
        params=params,
        device=device,
        noise_range=noise_range,
    )

    test_data = build_dataset(
        seeds=test_seeds,
        params=params,
        device=device,
        noise_range=noise_range,
    )

    return train_data, test_data