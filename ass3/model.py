import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch



"""
    构造上采样模块--左边特征提取基础模块    
"""

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
          
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


"""
    构造下采样模块--右边特征融合基础模块    
"""

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

"""
    模型主架构
"""

class UNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()


        n1 = 64
        filters = [n1, n1 * 2, n1 * 4]


        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #input x:[10,3,128,128]
        e1 = self.Conv1(x)
        #e1 [10, 64, 128, 128]
        e2 = self.Maxpool1(e1)  
        #e2 [10, 64, 64, 64])
        e2 = self.Conv2(e2)
        #e2 [10, 128, 64, 64]
        e3 = self.Maxpool2(e2)
        #e3 [10, 128, 32, 32]
        e3 = self.Conv3(e3)        
        #e3 [10, 256, 32, 32]
        d3 = self.Up3(e3)
        #d3 [10, 128, 64, 64]
        d3 = torch.cat((e2, d3), dim=1)       
        #d3 [10, 256, 64, 64]
        d3 = self.Up_conv3(d3)
        #d3 [10, 128, 64, 64]
        d2 = self.Up2(d3)
        #d2 [10, 64, 128, 128] 
        d2 = torch.cat((e1, d2), dim=1)

        #d2 [10, 128, 128, 128]
        d2 = self.Up_conv2(d2)
        #d2 [10, 64, 128, 128]
        out = self.Conv(d2)
        #out [10, 1, 128, 128]


        return out
