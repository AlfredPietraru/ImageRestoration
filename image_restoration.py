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

        self.nearest1 = nn.UpsamplingNearest2d(512, scale_factor=2)
        self.nearest2 = nn.UpsamplingNearest2d(512, scale_factor=2)
        self.nearest3 = nn.UpsamplingNearest2d(512, scale_factor=2)
        self.nearest4 = nn.UpsamplingNearest2d(512, scale_factor=2)
        self.nearest5 = nn.UpsamplingNearest2d(512, scale_factor=2)
        self.nearest6 = nn.UpsamplingNearest2d(256, scale_factor=2)
        self.nearest7 = nn.UpsamplingNearest2d(128, scale_factor=2)
        self.nearest8 = nn.UpsamplingNearest2d(64, scale_factor=2)

        self.concat1 = torch.cat((self.nearest1, self.enc_1), dim=1)
        self.concat2 = torch.cat((self.nearest2, self.enc_2), dim=1)
        self.concat3 = torch.cat((self.nearest3, self.enc_3), dim=1)
        self.concat4 = torch.cat((self.nearest4, self.enc_4), dim=1)
        self.concat5 = torch.cat((self.nearest5, self.enc_5), dim=1)
        self.concat6 = torch.cat((self.nearest6, self.enc_6), dim=1)
        self.concat7 = torch.cat((self.nearest7, self.enc_7), dim=1)
        self.concat8 = torch.cat((self.nearest8, self.enc_8), dim=1)

        self.pconv_09 = CompleteEncLayer(512, 512, 3, 1, "leaky_relu", True)
        self.pconv_10 = CompleteEncLayer(512, 512, 3, 1, "leaky_relu", True)
        self.pconv_11 = CompleteEncLayer(512, 512, 3, 1, "leaky_relu", True)
        self.pconv_12 = CompleteEncLayer(512, 512, 3, 1, "leaky_relu", True)
        self.pconv_13 = CompleteEncLayer(512, 256, 3, 1, "leaky_relu", True)
        self.pconv_14 = CompleteEncLayer(512, 128, 3, 1, "leaky_relu", True)
        self.pconv_15 = CompleteEncLayer(512, 64, 3, 1, "leaky_relu", True)
        self.pconv_16 = CompleteEncLayer(512, 3, 3, 1, "null", False)

    def forward(self, input, mask):
        input, mask = self.enc_1(input, mask)
        input, mask = self.enc_2(input, mask)
        input, mask = self.enc_3(input, mask)
        input, mask = self.enc_4(input, mask)
        input, mask = self.enc_5(input, mask)
        input, mask = self.enc_6(input, mask)
        input, mask = self.enc_7(input, mask)
        input, mask = self.enc_8(input, mask)

        
        return 
        