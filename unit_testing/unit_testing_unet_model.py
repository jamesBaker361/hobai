import sys
import os
import torch
sys.path.append(os.getcwd())
from unet.unet_model import *
import unittest

class UNetModelTestCase(unittest.TestCase):
    def setUp(self):
        self.model= UNet()

    def test_call(self):
        batch_size=4
        for dim in [32,128,512]:
            x=torch.randn(batch_size, 3, dim, dim)
            y= self.model(x)
            assert x.size() == y.size(), 'inputs and outputs dont match for dim {}'.format(dim)

if __name__=='__main__':
    unittest.main() # run all tests