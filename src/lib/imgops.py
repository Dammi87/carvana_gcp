from PIL import Image
import numpy as np

def load_image(img_path, size=None, interpolation=Image.ANTIALIAS):
    basewidth = 300
    img = Image.open(img_path)
    if size:
        img = img.resize(size, Image.ANTIALIAS)
    arr = np.array(img)
    
    arr_shape = arr.shape
    if len(arr.shape) == 2:
        arr_shape = (arr_shape[0], arr_shape[1], 1)
    return arr, arr_shape