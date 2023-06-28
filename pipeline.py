from typing import List
from transformation import Transformation


class Pipeline:
    def __init__(self, transformations: List[Transformation]):
        self.transformations=transformations