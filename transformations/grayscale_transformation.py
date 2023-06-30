from torch import tensor
from torchvision.transforms import Grayscale
from transformations.transformation import Transformation

class GrayscaleTransformation(Transformation):

    def __call__(self, patches_tensor: tensor, patch_size: int) -> tensor:
        return Grayscale()(patches_tensor)