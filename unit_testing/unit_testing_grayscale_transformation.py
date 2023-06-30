import sys
import os
import torch
sys.path.append(os.getcwd())
from pipeline import *
from transformations.grayscale_transformation import *
from testing_utils import *
import unittest

class GrayscaleTransformationTestCase(unittest.TestCase):
    def setUp(self):
        self.pipeline=Pipeline([GrayscaleTransformation()], [64])
        self.img=Image.open('cat.png')

    def test_call(self):
        transformed_img=self.pipeline(self.img)[0]
        transformed_img.save(UNIT_TESTING_OUTPUT_DIR+'/test_grayscale_call.png')

if __name__=='__main__':
    unittest.main() # run all tests