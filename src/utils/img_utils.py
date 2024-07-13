import torch
import cv2
import numpy as np

# np.random.seed(0)

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
  