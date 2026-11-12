import numpy as np
import logging
import os
import sys
import random
import torch
from datetime import datetime
import matplotlib.pyplot as plt

def get_logger_simple(file_dir, exp_name):
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
        print(f'Establish Log, Log_dir:{file_dir}')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    file_path = os.path.join(file_dir, f'{exp_name}_{timestamp}.log')
    file_handler = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"Logging initialized. Log file: {file_path}")


    return logger


def set_random_seed(seed: int):
    """Set random seeds for reproducibility."""
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def paint_spectrum(spectrum, save_path=None):

    spectrum = spectrum.numpy().reshape(90, 360)
    plt.imsave(save_path, spectrum, cmap='jet')
    spectrum = np.flipud(spectrum)
    # create a polar grid
    r = np.linspace(0, 1, 91) # change this depending on your radial distance
    theta = np.linspace(0, 2.*np.pi, 361)

    r, theta = np.meshgrid(r, theta)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    cax = ax.pcolormesh(theta, r, spectrum.T, cmap='jet', shading='flat')
    ax.axis('off')

    # save the image as a PNG file
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)




