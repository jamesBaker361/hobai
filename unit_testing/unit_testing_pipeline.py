import sys
import os
import torch
sys.path.append(os.getcwd())
from pipeline import *
import unittest

UNIT_TESTING_OUTPUT_DIR='unit_testing_output'

def are_images_identical(image1: Image.Image, image2: Image.Image) -> bool:
    # Compare image sizes
    if image1.size != image2.size:
        return False

    # Compare image modes
    if image1.mode != image2.mode:
        return False

    # Compare pixel data
    pixel_data1 = list(image1.getdata())
    pixel_data2 = list(image2.getdata())

    return pixel_data1 == pixel_data2

class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.patch_sizes=[32,64,128,256]
        self.pipeline=Pipeline([Transformation()],self.patch_sizes)
        self.img=Image.open('cat.png')

    def test_pad_img(self):
        img=self.img.resize((min(self.patch_sizes)-1,max(self.patch_sizes)+1))
        for p in self.patch_sizes:
            padded_img=self.pipeline.pad_img(img, p)
            width,height=padded_img.size
            padded_img.save(UNIT_TESTING_OUTPUT_DIR+ '/test_pad_img_{}.png'.format(p))
            assert width%p==0 and height%p==0, 'incorrectly resized image to {} x {}'.format(width,height)

    def test_patch_and_transform(self):
        img=self.img.resize((max(self.patch_sizes), max(self.patch_sizes)))
        for p in self.patch_sizes:
            recombined_img=self.pipeline.patch_and_transform(img, p)
            recombined_img.save(UNIT_TESTING_OUTPUT_DIR+ '/test_patch_and_transform_img_{}.png'.format(p))
            assert are_images_identical(img, recombined_img), 'recombined images are not identical'


    def test_call(self):
        img=self.img
        transformed_img_list = self.pipeline(img)
        for transformed_i in transformed_img_list:
            assert are_images_identical(img, transformed_i), 'recombined images are not identical'



if __name__=='__main__':
    unittest.main() # run all tests