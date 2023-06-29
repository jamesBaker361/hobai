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
        self.pipeline=Pipeline([CannySketchTransformation()], [64])
        self.img=Image.open('cat.png')

    def test_call(self):
        transformed_img=self.pipeline(self.img)[0]
        transformed_img.save(UNIT_TESTING_OUTPUT_DIR+'/test_canny_sketch_call.png')

if __name__=='__main__':
    unittest.main() # run all tests