import torch

class PSNR:
    def __init__(self, data_range=1.0):
        """Initialization of PSNR metric

        Args:
            data_range (float, optional): The data range of the target image 
            (distance between minimum and maximum possible values). Defaults to 1.0.
        """
        self.data_range = data_range

    def compute(self, output, target):
        mse = torch.mean((output - target)**2)
        psnr = 10 * torch.log10(self.data_range**2 / mse)
        return psnr
    
    def __str__(self):
        return f"PSNR data_range: {self.data_range}"