import sys
import os
import torch
sys.path.append(os.getcwd())
from pipeline import *
from canny_sketch_transformation import *
from testing_utils import *
import unittest

class CannySketchTransformationTestCase(unittest.TestCase):
    def setUp(self):
        self.patch_sizes=[32,64,256]
        self.pipeline=Pipeline([CannySketchTransformation()], self.patch_sizes)
        self.img=Image.open('cat.png')

    def test_call(self):
        for x,p_size in enumerate(self.patch_sizes):
            transformed_img=self.pipeline(self.img)[x]
            transformed_img.save(UNIT_TESTING_OUTPUT_DIR+'/test_canny_sketch_call_{}.png'.format(p_size))

if __name__=='__main__':
    unittest.main() # run all tests