import sys
import os
import torch
sys.path.append(os.getcwd())
from pipeline import *
import unittest

class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.pipeline=Pipeline([],[])
        self.patch_sizes=[32,64,128,256]

    def test_pad_img(self):
        img=Image.open('cat.png')
        img=img.resize((min(self.patch_sizes)-1,max(self.patch_sizes)+1))
        for p in self.patch_sizes:
            padded_img=self.pipeline.pad_img(img, p)
            width,height=padded_img.size
            assert width%p==0 and height%p==0, 'incorrectly resized image to {} x {}'.format(width,height)

if __name__=='__main__':
    unittest.main() # run all tests