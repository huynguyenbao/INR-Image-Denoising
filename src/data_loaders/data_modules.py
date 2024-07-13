import torch
from ..utils.img_utils import read_image, normalize, add_Gaussian_noise, image_numpy2torch

class SingleImageDataset:
    """
    Base class for all data loaders
    """
    def __init__(self, image_path, noise_level, RGB_mode, irange):
        input_image = read_image(image_path, RGB_mode)
        orange = [0, 255]
        
        noisy_image = normalize(add_Gaussian_noise(input_image, noise_level), orange, irange)
        clean_image = normalize(input_image, orange, irange)
        
        H, W = noisy_image.shape[:2]
        self.coords = self.get_mgrid(H, W)
        self.gt_noisy = image_numpy2torch(noisy_image, RGB_mode)
        self.gt_clean = image_numpy2torch(clean_image, RGB_mode)
        
        self.image_res = H * W
        self.image_shape = input_image.shape

    def get_mgrid(self, H, W):
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        
        X, Y = torch.meshgrid(x, y, indexing='xy')
        coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
        return coords
