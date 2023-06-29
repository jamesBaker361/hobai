from PIL import Image

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