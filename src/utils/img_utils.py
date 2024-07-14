import torch
import cv2
import numpy as np

np.random.seed(0)

def add_Gaussian_noise(image, sigma):
    gaussian_noise = np.random.randn(*image.shape) * sigma
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.float32)
    return noisy_image

def read_image(image_path, RGB_mode):
    if RGB_mode == False:
        img = cv2.imread(image_path, 0).astype(np.float32)
    else:
        img = cv2.imread(image_path)[...,::-1].astype(np.float32)
    return img

def save_image(image_path, image, RGB_mode):
    if RGB_mode == False:
        cv2.imwrite(image_path, image)
    else:
        cv2.imwrite(image_path, image[...,::-1])
      
def normalize(x, range1, range2):
    """Normailize input from range1 to range2
    """
    min1 = range1[0]
    max1 = range1[1]

    min2 = range2[0]
    max2 = range2[1]

    aux = (max2 - min2) / (max1 - min1)
    return (x - min1) * aux + min2

def image_numpy2torch(image, RGB_mode):
    H, W = image.shape[:2]
    c = 3 if RGB_mode else 1
    return torch.tensor(image).reshape(H*W, c)[None, ...]
  
def create_image_pyramid(image, levels):
    """Create Image Pyramid

    Args:
        image (_type_): [0, 255]
        levels (_type_): number of layers
    """
    pyramid = np.zeros((*image.shape, levels), dtype=np.float32)
    pyramid[...,0] = image
    
    down = image

    for i in range(1, levels):
        H_d, W_d = down.shape[:2]
        down = cv2.resize(down, (H_d // 2, W_d // 2))

        up = down
        for _ in range(i):
            H_u, W_u = up.shape[:2]
            up = cv2.resize(up, (H_u * 2, W_u * 2), interpolation=cv2.INTER_NEAREST)
        
        pyramid[...,i] = up
    
    return pyramid

        