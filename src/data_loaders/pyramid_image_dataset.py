import torch
from ..utils.img_utils import read_image, normalize, add_Gaussian_noise, image_numpy2torch, create_image_pyramid
from ..utils.noise_estimation import noise_estimate

class PyramidImageDataset:
    """
    Base class for all data loaders
    """
    def __init__(self, image_path, noise_level, pyramid_levels, RGB_mode, target_range, **dasetset_kwargs):

        # Prepare Images
        input_image = read_image(image_path, RGB_mode)
        corrupted_image = add_Gaussian_noise(input_image, noise_level)
        pyramid = create_image_pyramid(corrupted_image, pyramid_levels)

        # Estimate Noise Level
        self.noise_level_est = noise_estimate(corrupted_image)

        # Intensity Transformation
        orange = [0, 255]
        noisy_images = normalize(pyramid, orange, target_range)
        clean_image = normalize(input_image, orange, target_range)
        
        # Create Training Data
        H, W = clean_image.shape[:2]
        self.coords = self.get_mgrid(H, W)
        self.gt_noisy = torch.stack([
            image_numpy2torch(noisy_images[...,i], RGB_mode) for i in range(pyramid_levels)
            ], axis=0)
        self.gt_clean = image_numpy2torch(clean_image, RGB_mode)
        
        self.image_res = H * W
        self.image_shape = input_image.shape
        self.pyramid_levels = pyramid_levels

    def get_mgrid(self, H, W):
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        
        X, Y = torch.meshgrid(x, y, indexing='xy')
        coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
        return coords
