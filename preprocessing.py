import os
import cv2
import torch

ORIGINAL_TRAINING_IMAGES_PATH = './images/train/groundtruth/'
MODIFIED_TRAINING_IMAGES_PATH = './images/train/modified/'

def get_path_training(PATH : str, f : str):
        return PATH + f

def get_new_path(PATH : str, idx : int):
        return PATH + "image" + str(100 + idx) + ".png"

def get_image_color(i : int):
    if i < 0 | i > 90:
          print("wrong number taken")
          exit(1)
    img = cv2.imread(get_new_path(MODIFIED_TRAINING_IMAGES_PATH, i),
                      cv2.IMREAD_COLOR)
    R,G,B = cv2.split(img)
    img = torch.Tensor(img)
    img = img.permute((2, 0, 1))
    img = img.unsqueeze(0)
    mask = R + G + B 
    mask = torch.Tensor(mask) > 0
    mask = torch.cat((mask, mask, mask), dim=1)
    mask = mask.reshape(shape=(512, 512, 3))
    mask = mask.permute((2, 0, 1))
    mask = mask.unsqueeze(0)
    return torch.Tensor(img).to(torch.float), mask.to(torch.uint8)


def rename_folder_images_path(PATH : str, NEW_PATH):
    files = os.listdir(PATH)
    files.sort()
    for (idx, f) in enumerate(files):
        if (idx % 2 == 1):
            os.rename(get_path_training(PATH, f), get_new_path(NEW_PATH, idx))
        else:
            os.rename(get_path_training(PATH, f), get_new_path(PATH, idx))