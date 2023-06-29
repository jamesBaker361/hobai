from torch import tensor

class Transformation:
    def __init__(self):
        pass

    def __call__(self, patches_tensor: tensor, patch_size: int) -> tensor:
        return patches_tensor