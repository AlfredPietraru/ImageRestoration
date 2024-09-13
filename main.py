import torch
from torchvision.transforms.functional import rotate
import preprocessing as prep
from net import AutoEncoder
from torch.utils.data import  DataLoader
import os
LEARNING_RATE = 1e-3
EPOCHS = 3

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

def transformers(img : torch.tensor):
  return torch.cat([rotate(img, angle=90),
      rotate(img, angle=180), rotate(img, angle=270)], dim=0)


def training_loop(dataloader : DataLoader):
    model = AutoEncoder()
    if (os.path.isfile("./reconstruction.pth")):
      model.load_state_dict(torch.load("./reconstruction.pth"))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    for i in range(EPOCHS):
      for _, img_info in  enumerate(dataloader):
        img, modified, _ = img_info
        for i in range(0, 4):
          if (i == 0):
             loss =loss_fn(model(modified), img)
          if (i != 0):
            loss = loss_fn(model(rotate(modified, i * 90)), rotate(img, i * 90))
          optimizer.zero_grad()
          print(loss)
          loss.backward()
          optimizer.step()
      print("gata epoca %d", i)
      torch.save(model.state_dict(), "./reconstruction.pth")
        

def eval_loop(idx : int):
   model = AutoEncoder()
   model.load_state_dict(torch.load("./reconstruction.pth"))
   img, mask = prep.get_test_image(idx)
   out = model(img)
   prep.show_image(out.squeeze(dim=0))
   final_image = img + out * mask
   prep.show_image(final_image.squeeze(dim=0))

images = ImageDataset()
dataloader = DataLoader(images, batch_size=5, shuffle=True)

training_loop(dataloader)
eval_loop(2)