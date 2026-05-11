import numpy as np
import torch

from NN_tomo.Algo_setup_tomo import get_setup, build_algo_functions,ray


def load_sample(A,AT,space, seed, params, device, noise_range=(0.05, 0.2)):
    """
    Build one sample:
        noisy measurement (y) / clean image (x_true) / algorithmic functions
    """

    
    noise_level = np.random.uniform(*noise_range)


    setup = get_setup(A,AT,space, seed=seed, noise_level=noise_level, device=device)


    functions = build_algo_functions(setup, params)


    init_state = setup["initial_state"].to(device)        
    clean = setup["x_true"].to(device)   
  
    return init_state, clean, functions


def build_dataset(size, seeds, params, device, noise_range=(0.05, 0.2)):

    ra=ray(device,size)
    A=ra["A"]
    AT=ra["AT"]
    space=ra["space"]
    
    data = [
        load_sample(
            A,AT,space,
            seed=s,
            params=params,
            device=device,
            noise_range=noise_range,
        )
        for s in seeds
    ]

    return data


def split_seeds(train_seeds, test_seeds):
    return list(train_seeds), list(test_seeds)


def build_train_test_data(train_seeds, test_seeds, params, device, noise_range=(0.05, 0.2)):

    size = params.size

    train_data = build_dataset(
        size=size,
        seeds=train_seeds,
        params=params,
        device=device,
        noise_range=noise_range,
    )

    test_data = build_dataset(
        size=size,
        seeds=test_seeds,
        params=params,
        device=device,
        noise_range=noise_range,
    )

    return train_data, test_data