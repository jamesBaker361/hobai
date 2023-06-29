from typing import Any, List
from PIL import Image
from torch import tensor,stack
from transformation import Transformation
from torchvision.transforms import ToTensor, ToPILImage


class Pipeline:
    def __init__(self, transformation_list: List[Transformation], patch_size_list: List[int]):
        self.transformation_list=transformation_list
        self.patch_size_list=patch_size_list

    def __call__(self, img: Image.Image) -> List[Image.Image]:
        transformed_img_list=[]
        for p in self.patch_size_list:
            padded_img=self.pad_img(img=img, patch_size=p)
            transformed_img=self.patch_and_transform(img=padded_img, patch_size=p)
            transformed_img_list.append(transformed_img)

        return transformed_img_list

    def pad_img(self, img: Image.Image, patch_size: int) -> List[Image.Image]:
        width, height = img.size

        # Calculate the amount of padding needed
        pad_width = (width // patch_size + 1) * patch_size - width
        pad_height = (height // patch_size + 1) * patch_size - height

        # Create a new blank image with the desired dimensions
        padded_img = Image.new(img.mode, (width + pad_width, height + pad_height), color='white')

        # Paste the original image onto the padded image
        padded_img.paste(img, (0, 0))

        return padded_img

    def patch_and_transform(self, img: Image.Image, patch_size: int) -> List[Image.Image]:
        patches = []
        width, height = img.size

        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                patch = img.crop((x, y, x + patch_size, y + patch_size))
                patches.append(patch)

        patches_tensor=stack([ToTensor()(p) for p in patches])
        for transformation in self.transformation_list:
            patches_tensor=transformation(patches_tensor, patch_size=patch_size)

        new_patches = [ToPILImage()(patches_tensor[i]) for i in range(patches_tensor.size(0))]

        recombined_img = Image.new("RGB", (width, height))

        patch_index = 0
        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                transformed_patch = new_patches[patch_index]
                recombined_img.paste(transformed_patch, (x, y))
                patch_index += 1

        return recombined_img