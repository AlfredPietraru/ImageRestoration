import torch
import os
import preprocessing as prep
MODIFIED_TRAINING_IMAGES_PATH = './images/train/modified/'

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.root_dir = MODIFIED_TRAINING_IMAGES_PATH

    def __len__(self):
        elements  = os.listdir(self.root_dir)
        return len(elements)

    def __getitem__(self, idx):
        if (idx < 0 | idx > self.__len__()):
            return None
        return prep.get_full_training_image(idx)