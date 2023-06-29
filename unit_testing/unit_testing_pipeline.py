import sys
import os
import torch
sys.path.append(os.getcwd())
from pipeline import *
from testing_utils import *
import unittest

UNIT_TESTING_OUTPUT_DIR='unit_testing_output'

class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.patch_size_list=[32,64,128,256]
        self.pipeline=Pipeline([Transformation()],self.patch_size_list)
        self.img=Image.open('cat.png')

    def test_pad_img(self):
        img=self.img.resize((min(self.patch_size_list)-1,max(self.patch_size_list)+1))
        for p in self.patch_size_list:
            padded_img=self.pipeline.pad_img(img, p)
            width,height=padded_img.size
            padded_img.save(UNIT_TESTING_OUTPUT_DIR+ '/test_pad_img_{}.png'.format(p))
            assert width%p==0 and height%p==0, 'incorrectly resized image to {} x {}'.format(width,height)

    def test_patch_and_transform(self):
        img=self.img.resize((max(self.patch_size_list), max(self.patch_size_list)))
        for p in self.patch_size_list:
            recombined_img=self.pipeline.patch_and_transform(img, p)
            recombined_img.save(UNIT_TESTING_OUTPUT_DIR+ '/test_patch_and_transform_img_{}.png'.format(p))
            assert are_images_identical(img, recombined_img), 'recombined images are not identical'


    def test_call(self):
        img=self.img
        transformed_img_list = self.pipeline(img)
        for transformed_i in transformed_img_list:
            assert are_images_identical(img, transformed_i), 'images are not identical'



if __name__=='__main__':
    unittest.main() # run all tests