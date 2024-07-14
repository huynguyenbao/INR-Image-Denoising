import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_image_pyramid(image, levels):
    """Create Image Pyramid

    Args:
        image (_type_): [0, 255]
        levels (_type_): number of layers
    """
    pyramid = np.zeros((*image.shape, levels))
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

im = cv2.imread('data/cameraman.png', 0).astype(np.float32)

n = 4
pyramid = create_image_pyramid(im, n)

im2 = np.concatenate([pyramid[...,i] for i in range(n)], axis=-1)
plt.imshow(im2, cmap='gray')
plt.show()        