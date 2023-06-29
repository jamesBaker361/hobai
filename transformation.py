from torch import tensor

class Transformation:
    def __init__(self):
        pass

    def __call__(self,patch: tensor, patch_size: int) -> tensor:
        return patch