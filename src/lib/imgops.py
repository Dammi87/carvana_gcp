from PIL import Image
import numpy as np

def load_image(img_path, size=None, scale=None, interpolation=Image.ANTIALIAS):
    img = Image.open(img_path)
    if scale:
        image_size = img.size
        size = tuple([int(dim * scale) for dim in image_size])

    if size:
        img = img.resize(size, Image.ANTIALIAS)
    arr = np.array(img)
    
    arr_shape = arr.shape
    if len(arr.shape) == 2:
        arr_shape = (arr_shape[0], arr_shape[1], 1)
    return arr, arr_shape