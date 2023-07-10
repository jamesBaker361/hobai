import sys
import os
import torch
sys.path.append(os.getcwd())
from pipeline import *
from transformations.grayscale_transformation import *
from testing_utils import *
import imageio
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
        img=self.img.resize((100,100))
        transformed_img_list = self.pipeline(img)
        for transformed_i,p in zip(transformed_img_list, self.pipeline.patch_size_list):
            transformed_i.save(UNIT_TESTING_OUTPUT_DIR+ '/test_call_{}.png'.format(p))

    def test_get_incremental_list(self):
        dims=[64,128]
        incremental_pipeline=Pipeline([GrayscaleTransformation()],dims)
        incremental_pipeline(self.img)
        incrementals = incremental_pipeline.get_incremental_list()
        for x in range(len(dims)):
            incremental=incrementals[x]
            output_path='unit_testing_output/incremental_call_{}.gif'.format(dims[x])
            # Save the images as frames of the GIF
            with imageio.get_writer(output_path, mode='I') as writer:
                for img in incremental:
                    writer.append_data(img)

        



if __name__=='__main__':
    unittest.main() # run all tests