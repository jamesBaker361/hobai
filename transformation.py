from torch import tensor

class Transformation:
    def __init__(self):
        pass

    def transform(self,patch: tensor, x: int, y: int) -> tensor:
        return patch