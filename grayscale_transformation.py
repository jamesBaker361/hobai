from torch import tensor
from torchvision.transforms import Grayscale
from transformation import Transformation

class GrayscaleTransformation(Transformation):

    def __call__(self,patch: tensor, patch_size: int) -> tensor:
        return Grayscale()(patch)