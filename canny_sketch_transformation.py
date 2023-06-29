from torch import tensor, from_numpy, stack
import numpy as np
from transformation import Transformation
import cv2

class CannySketchTransformation(Transformation):
    def __init__(self, t_lower: int =50, t_upper: int=200, aperture_size: int =7, L2gradient: bool = True):
        self.t_lower=t_lower
        self.t_upper=t_upper
        self.aperture_size=aperture_size
        self.L2gradient=L2gradient
    
    def __call__(self, patches_tensor: tensor, patch_size: int) -> tensor:
        # Convert the batch tensor to a numpy array
        batch_np = np.transpose(patches_tensor.numpy(), (0, 2, 3, 1))

        # Convert the numpy array to grayscale images
        batch_gray = np.asarray([255*cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) for image_np in batch_np]).astype(np.uint8)

        # Apply Canny edge detection to each grayscale image
        batch_edges = np.asarray([cv2.Canny(image_gray, 
                                            self.t_lower,
                                            self.t_upper,
                                            self.aperture_size,
                                            L2gradient = self.L2gradient) for image_gray in batch_gray])

        # Convert the edges array back to a tensor
        edges_tensor = from_numpy(batch_edges).float()

        # Normalize the tensor values to [0, 1]
        edges_tensor = edges_tensor/255
        # Invert
        edges_tensor = 1- edges_tensor

        # Convert to 3-D tensor
        return stack([edge.expand(3,-1,-1) for edge in edges_tensor])