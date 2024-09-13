import torch
import preprocessing as prep
from net import AutoEncoder
from torch.utils.data import  DataLoader
from load_in_memory import ImageDataset
LEARNING_RATE = 1e-3
EPOCHS = 5

def training_loop(dataloader : DataLoader):
    model = AutoEncoder()
    model.load_state_dict(torch.load("./reconstruction.pth"))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    for i in range(EPOCHS):
      for _, img_info in  enumerate(dataloader):
        img, modified, _ = img_info
        modified = modified.squeeze(dim=0)
        img = img.squeeze(dim=0)
        loss = loss_fn(model(modified), img)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      print("gata epoca %d", i)
      torch.save(model.state_dict(), "./reconstruction.pth")
        

def eval_loop():
   model = AutoEncoder()
   model.load_state_dict(torch.load("./reconstruction.pth"))
   img, modified, _ =prep.get_full_training_image(87)
   out = model(modified)
   prep.show_image(out)

images = ImageDataset()
dataloader = DataLoader(images, batch_size=1, shuffle=True)

training_loop(dataloader)
eval_loop()