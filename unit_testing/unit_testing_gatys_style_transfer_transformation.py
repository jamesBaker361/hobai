import sys
import os
import torch
sys.path.append(os.getcwd())
from pipeline import *
from transformations.gatys_style_transfer_transformation import *
from testing_utils import *
import unittest
from torchvision.transforms import ToTensor, ToPILImage

class GatysStyleTransferTransformationTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_tf_to_pytorch(self):
        batch_size = 1  # Number of imgs in the batch
        channels = 3  # Number of color channels (e.g., 3 for RGB)
        width = 256  # Width of the image
        height = 256  # Height of the image

        tf_imgs=tf.random.normal((batch_size,width,height,channels))

        torch_imgs=tf_to_pytorch(tf_imgs)

        assert torch_imgs.numpy().shape==(batch_size, channels, width, height)

    def test_pytorch_to_tf(self):
        batch_size = 2  # Number of imgs in the batch
        channels = 3  # Number of color channels (e.g., 3 for RGB)
        width = 256  # Width of the image
        height = 256  # Height of the image

        torch_imgs = torch.randn(batch_size, channels, width, height)

        tf_imgs=pytorch_to_tf(torch_imgs)

        assert tf_imgs.numpy().shape==(batch_size,width,height,channels)

    def test_two_way_translation(self):
        img=Image.open('robert.jpg')
        torch_imgs=ToTensor()(img)
        torch_imgs=torch_imgs.unsqueeze(0)
        tf_imgs=pytorch_to_tf(torch_imgs)
        reconstructed_torch_imgs=tf_to_pytorch(tf_imgs)
        reconstructed_img=ToPILImage()(reconstructed_torch_imgs[0])
        reconstructed_img.save('new_robert.jpg')
        assert reconstructed_img.size==img.size
        


    def test_call(self):
        gatys_style_transfer_transformation = GatysStyleTransferTransformation("cat.png")
        dims=[32,64,128,256,512]
        pipeline = Pipeline([gatys_style_transfer_transformation], dims)
        content_img=Image.open('robert.jpg')
        for x,dim in enumerate(dims):
            transformed_img=pipeline(content_img)[x]
            transformed_img.save(UNIT_TESTING_OUTPUT_DIR+'/test_gatys_style_transfer_call_{}.png'.format(dim))



if __name__=='__main__':
    unittest.main() # run all tests