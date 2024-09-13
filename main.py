import torch
import torchvision.transforms.functional
import preprocessing as prep
from net import AutoEncoder
from torch.utils.data import  DataLoader
import torchvision
import os
LEARNING_RATE = 1e-3
EPOCHS = 1

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.root_dir = prep.MODIFIED_TRAINING_IMAGES_PATH

    def __len__(self):
        elements  = os.listdir(self.root_dir)
        return len(elements)

    def __getitem__(self, idx):
        if (idx < 0 | idx > self.__len__()):
            return None
        return prep.get_full_training_image(idx)

def training_loop(dataloader : DataLoader):
    model = AutoEncoder()
    model.load_state_dict(torch.load("./reconstruction.pth"))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    for i in range(EPOCHS):
      for _, img_info in  enumerate(dataloader):
        img, modified, _ = img_info
        loss = loss_fn(model(modified), img)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      print("gata epoca %d", i)
      torch.save(model.state_dict(), "./reconstruction.pth")
        

def eval_loop(idx : int):
   model = AutoEncoder()
   model.load_state_dict(torch.load("./reconstruction.pth"))
   img, mask = prep.get_test_image(idx)
   final_image = img + model(img) * mask
   prep.show_image(final_image.squeeze(dim=0))
   rotated =torchvision.transforms.functional.rotate(final_image, 90)

images = ImageDataset()
dataloader = DataLoader(images, batch_size=1, shuffle=True)

# training_loop(dataloader)
eval_loop(2)