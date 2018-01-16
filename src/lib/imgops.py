from PIL import Image
import numpy as np

def load_image(img_path, size=None, interpolation=Image.ANTIALIAS):
    basewidth = 300
    img = Image.open(img_path)
    if size:
        img = img.resize(size, Image.ANTIALIAS)
    
    return np.array(img)