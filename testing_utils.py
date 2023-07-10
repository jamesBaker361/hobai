from PIL import Image

UNIT_TESTING_OUTPUT_DIR='unit_testing_output'

def are_images_identical(image1: Image.Image, image2: Image.Image) -> bool:
    '''The function `are_images_identical` compares two images to determine if they are identical based on
    their size, mode, and pixel data.
    
    Parameters
    ----------
    image1 : Image.Image
        The parameter `image1` is of type `Image.Image`, which suggests that it is an image object. It is
    likely that this parameter represents the first image that needs to be compared.
    image2 : Image.Image
        The `image2` parameter is an instance of the `Image.Image` class. It represents the second image
    that you want to compare with `image1`.
    
    Returns
    -------
        a boolean value indicating whether the two input images are identical or not.
    
    '''
    if image1.size != image2.size:
        return False

    # Compare image modes
    if image1.mode != image2.mode:
        return False

    # Compare pixel data
    pixel_data1 = list(image1.getdata())
    pixel_data2 = list(image2.getdata())

    return pixel_data1 == pixel_data2