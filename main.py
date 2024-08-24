import torch
from image_restoration import ImageRestoration
import preprocessing as prep

img, mask = prep.get_image_color(0)
model = ImageRestoration(img.shape[0])
print(model(img, mask))