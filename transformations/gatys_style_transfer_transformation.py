from transformations.transformation import Transformation
from PIL import Image
import torch
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from torch import tensor, from_numpy, stack

def pytorch_to_tf(torch_images):

    # Permute dimensions to change the order of axes
    torch_images_permuted = torch_images.permute(0, 2, 3, 1)

    # Convert PyTorch tensor to NumPy array
    numpy_images = torch_images_permuted.numpy()

    # Convert NumPy array to TensorFlow tensor
    tf_images = tf.convert_to_tensor(numpy_images, dtype=tf.float32)

    return tf_images

def tf_to_pytorch(tf_images):
    # Convert TensorFlow tensor to NumPy array
    numpy_images = tf_images.numpy()

    # Convert NumPy array to PyTorch tensor
    torch_images = torch.from_numpy(numpy_images)

    # Permute dimensions to change the order of axes
    torch_images = torch_images.permute(0, 3, 1, 2)

    return torch_images

def get_img(path: str):
    # Load the image using PIL
    with Image.open(path) as image:
        # Convert the image to a numpy array
        image_array = np.array(image).astype(np.float32)/255

    # Convert the numpy array to a TensorFlow tensor
    tensor_image = tf.convert_to_tensor(image_array)

    tensor_image=tf.expand_dims(tensor_image, axis=0)

    return tf.image.resize_with_pad(tensor_image, *(256, 256))

class GatysStyleTransferTransformation(Transformation):
    def __init__(self, style_image_path: str):
        self.style_image_path=style_image_path
        self.style_image=get_img(style_image_path)
        self.hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    def __call__(self, patches_tensor: tensor, patch_size: int) -> tensor:
        tf_patches_tensor=pytorch_to_tf(patches_tensor)
        tf_patches_tensor_stylized = self.hub_module(tf.constant(tf_patches_tensor), tf.constant(self.style_image))[0]
        patches_tensor_stylized=tf_to_pytorch(tf_patches_tensor_stylized)
        return patches_tensor_stylized