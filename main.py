import torch
from image_restoration import ImageRestoration
from net import PConvUNet
import preprocessing as prep

def train_loop(model, optimizer, loss_function):
    model = PConvUNet()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for k in range(1, 10, 1):
        for i in range(1, 4, 1):
            original, altered, mask = prep.get_info_image_training(i)
            output, mask = model(altered, mask)
            loss = loss_function(original, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
            print("a terminat o imagine")
        print("gata o epoca")
    torch.save(model.state_dict(), "./model.pth")

def evaluation(idx : int):
    model = PConvUNet()
    model.load_state_dict(torch.load("./model.pth"))
    model.eval
    original, altered, mask = prep.get_info_image_training(idx)
    output, mask = model(altered, mask)
    prep.debug_image(output)
    
    
# train_loop(model, optimizer, loss_function)
evaluation(4)