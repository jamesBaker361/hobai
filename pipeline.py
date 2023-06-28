from typing import Any, List
from PIL import Image
from transformation import Transformation


class Pipeline:
    def __init__(self, transformations: List[Transformation], patch_sizes: List[int]):
        self.transformations=transformations
        self.patch_sizes=patch_sizes

    def __call__(self, img: Image) -> Any:
        pass

    def pad_img(self, img: Image, patch_size: int) -> Image:
        width, height = img.size

        # Calculate the amount of padding needed
        pad_width = (width // patch_size + 1) * patch_size - width
        pad_height = (height // patch_size + 1) * patch_size - height

        # Create a new blank image with the desired dimensions
        padded_img = Image.new(img.mode, (width + pad_width, height + pad_height), color='white')

        # Paste the original image onto the padded image
        padded_img.paste(img, (0, 0))

        return padded_img