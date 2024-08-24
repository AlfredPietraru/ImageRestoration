import torch
from torch import nn
import math

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, 0, 1, 1, True)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, 0, 1, 1, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            mask = mask.to(torch.float)
            output_mask = self.mask_conv(mask)
        no_update_holes = output_mask == 0
        
        mask_sum = output_mask.masked_fill_(no_update_holes, 1)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0).to(torch.uint8)

        return output, new_mask
    
class CompleteDecLayer(nn.Module):
    def __init__(self, nearest_filter : int, kernel_size = 0):
        super().__init__()
        self.upsample = nn.Upsample(size=nearest_filter)
        self.


class CompleteEncLayer(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int,
                 stride : int, activation_f : str, batch_normalization : bool):
        super().__init__()
        self.main_layer  = PartialConv(in_channels, out_channels,
                                        kernel_size, stride)
        self.bn = self.decide_on_batch_normalization(batch_normalization,
                                                      out_channels)
        self.activation_function = self.decide_on_activation(activation_f) 

    def decide_on_batch_normalization(self, value : bool, out_channels : int):
        if (value):
            return nn.BatchNorm2d(out_channels)
        return 

    def decide_on_activation(self, act_f : str):
        if (act_f == "relu"):
            return nn.ReLU()
        if (act_f == "leaky_relu"):
            return nn.LeakyReLU(0.2)
        return 

            
    def forward(self, input, mask):
        input, mask = self.main_layer(input, mask)
        if self.bn != None:
            input = self.bn(input)
        if self.activation_function != None:
            input = self.activation_function(input)
        return input, mask.to(torch.uint8)
            

class ImageRestoration(nn.Module):
    def __init__(self, image_size : int):
        super().__init__()
        self.enc_1 = CompleteEncLayer(3, 64, 7, 2, "null", False)
        self.enc_2 = CompleteEncLayer(64, 128, 5, 2, "relu", True)
        self.enc_3 = CompleteEncLayer(128, 256, 5, 2, "relu", True)
        self.enc_4 = CompleteEncLayer(256, 512, 3, 1, "relu", True)  # HERE MIGHT ARISE PROBLEMS
        self.enc_5 = CompleteEncLayer(512, 512, 3, 2, "relu", True)
        self.enc_6 = CompleteEncLayer(512, 512, 3, 2, "relu", True)
        self.enc_7 = CompleteEncLayer(512, 512, 3, 2, "relu", True)
        self.enc_8 = CompleteEncLayer(512, 512, 3, 2, "relu", True)

        self.dec9 = 
            
        self.flatten = nn.Flatten()

    def forward(self, input, mask):
        input, mask = self.enc_1(input, mask)
        input, mask = self.enc_2(input, mask)
        input, mask = self.enc_3(input, mask)
        input, mask = self.enc_4(input, mask)
        input, mask = self.enc_5(input, mask)
        input, mask = self.enc_6(input, mask)
        input, mask = self.enc_7(input, mask)
        return self.enc_8(input, mask)
        