import os
import cv2
import torch

INITIAL_IMAGES_PATH = "./images/train/initial_images/"
ORIGINAL_TRAINING_IMAGES_PATH = './images/train/groundtruth/'
MODIFIED_TRAINING_IMAGES_PATH = './images/train/modified/'

def map_int_to_string(idx : int):
     if idx >= 100:
          return str(idx)
     if idx <= 99 & idx >= 10:
          return "0" + str(idx)
     if idx >= 0 & idx <= 9:
          return "00" + str(idx)
     
def get_groundtruth_image_path(idx : int):
    name = "0000" + map_int_to_string(idx) + ".png"
    return ORIGINAL_TRAINING_IMAGES_PATH + name

def get_modified_image_path(idx : int):
    name = "m_0000" + map_int_to_string(idx) + ".png"
    return MODIFIED_TRAINING_IMAGES_PATH + name

def create_mask(img):
    R,G,B = cv2.split(img)
    mask = R + G + B 
    mask = torch.Tensor(mask) > 0
    mask = torch.cat((mask, mask, mask), dim=1)
    mask = mask.reshape(shape=(512, 512, 3))
    mask = mask.permute((2, 0, 1))
    mask = mask.unsqueeze(0)
    return mask.to(torch.uint8)

def debug_image(img : torch.Tensor):
    img = img.squeeze(0)
    img = img.permute((1, 2, 0))
    print(img.to(torch.uint8))
    cv2.imshow("Input", img.to(torch.uint8).detach().numpy())
    cv2.waitKey(0)

def change_matrix_shape(img):
    img = torch.Tensor(img)
    img = img.permute((2, 0, 1))
    img = img.unsqueeze(0)
    return torch.Tensor(img).to(torch.float)

def get_info_image_training(i : int):
    if i < 0 | i > 500:
          print("wrong number taken")
          exit(1)
    altered_image = cv2.imread(get_modified_image_path(i),
                      cv2.IMREAD_COLOR)
    original_image = cv2.imread(get_groundtruth_image_path(i),
                                 cv2.IMREAD_COLOR)
    return change_matrix_shape(torch.Tensor(original_image)), change_matrix_shape(altered_image), create_mask(altered_image)


def resize_images(input_folder, output_folder, size=(512, 512)):
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            resized_img = cv2.resize(img, size)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_img)


