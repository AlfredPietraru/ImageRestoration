import os
import cv2
import torch

INITIAL_IMAGES_PATH = './images/train/initial_images/'
GROUNDTRUTH_IMAGE_PATH = './images/train/groundtruth/'
MODIFIED_TRAINING_IMAGES_PATH = './images/train/modified/'


SIZE = 512
def map_int_to_string(idx: int):
    return f"{idx:03d}.png"

def create_initial_images(input_folder, output_folder, size=SIZE):
    for idx, filename in enumerate(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, (512, 512))
        output_path = os.path.join(output_folder, map_int_to_string(idx + 1))
        cv2.imwrite(output_path, resized_img)

def create_squares_coordinates(length = 20, square_numbers = 20):
    upperLeftPoint = torch.randint(0, SIZE - length, size=(square_numbers, 2))
    x_axis = upperLeftPoint[:, 0].reshape((square_numbers, 1))
    y_axis = upperLeftPoint[:, 1].reshape((square_numbers, 1))
    squares_sizes = torch.randint(10, length +1, size=(square_numbers, 1)) 
    lowerLeftPoint = torch.cat([x_axis + squares_sizes, y_axis], dim=1)
    upperRightPoint = torch.cat([x_axis, y_axis + squares_sizes], dim=1)
    lowerRightPoint = torch.cat([x_axis + squares_sizes, y_axis + squares_sizes], dim=1)
    return torch.cat([upperLeftPoint, upperRightPoint, lowerLeftPoint, lowerRightPoint], dim=1)

def show_matrix(matrix):
    matrix = matrix.to(torch.uint8)
    cv2.imshow("gata", (matrix * 255).detach().numpy())
    cv2.waitKey(0)

def create_mask(length, square_numbers):
    coordinates = create_squares_coordinates(length, square_numbers)
    vector = torch.tensor(range(0, SIZE))
    matrix = vector.repeat(SIZE).reshape((SIZE, SIZE))
    mask = torch.zeros(size=(SIZE, SIZE)).to(torch.bool)
    for xy in coordinates: 
        first_mask = torch.logical_and(matrix >= xy[0], matrix <= xy[4]).to(torch.uint8)
        second_mask = torch.logical_and(torch.transpose(matrix, 0, 1) >= xy[1],
                torch.transpose(matrix, 0, 1) <= xy[3]).to(torch.uint8)
        current_mask = torch.logical_and(first_mask, second_mask)
        mask = torch.logical_or(current_mask, mask)
    return torch.logical_not(mask)

def create_obscured_images(input_folder, output_folder):
  filenames = os.listdir(input_folder)
  filenames.sort()
  for idx, filename in enumerate(filenames):
    mask = create_mask(50, 50).unsqueeze(dim=-1)
    img_path = os.path.join(input_folder, filename)
    img = torch.tensor(cv2.imread(img_path))
    img = img * mask
    output_path = os.path.join(output_folder, map_int_to_string(idx + 1))
    cv2.imwrite(output_path, img.detach().numpy())


def get_full_training_image(idx : int):
    filename = map_int_to_string(idx)
    pure_image = cv2.imread(INITIAL_IMAGES_PATH + filename, cv2.IMREAD_COLOR)
    affected_image = cv2.imread(MODIFIED_TRAINING_IMAGES_PATH + filename, 
                                cv2.IMREAD_COLOR)
    pure_image = torch.tensor(pure_image).to(torch.float32).permute((2, 0, 1))
    affected_image = torch.tensor(affected_image)
    masks = affected_image == 0
    mask1, mask2, mask3 = torch.chunk(masks, 3,dim=2)
    mask = torch.bitwise_and(torch.bitwise_and(mask1, mask2), mask3)
    affected_image = affected_image.to(torch.float32).permute((2, 0, 1))
    return pure_image, affected_image, mask.squeeze(dim=-1) 

def get_only_training_image(batch_index : int, batch_size):
    images_indexes = range(batch_index * batch_size + 1, (batch_index + 1) * batch_size + 1) 
    filenames = list(map(map_int_to_string, images_indexes))
    images = torch.zeros(size=(batch_size, 3, SIZE, SIZE))
    for idx, filename in enumerate(filenames):
        img = cv2.imread(MODIFIED_TRAINING_IMAGES_PATH + filename, 
                                cv2.IMREAD_COLOR)
        images[idx] = torch.tensor(img).permute((2, 0, 1))
    return images


def create_training_data():
    create_initial_images(GROUNDTRUTH_IMAGE_PATH, INITIAL_IMAGES_PATH)
    create_obscured_images(INITIAL_IMAGES_PATH, MODIFIED_TRAINING_IMAGES_PATH)
