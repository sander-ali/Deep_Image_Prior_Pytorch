import torch.nn as nn
import torch

class network_arch(nn.Module):
    def __init__(self):
        super(network_arch, self).__init__()

        self.leaky_relu = nn.LeakyReLU()

        self.conv_1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.bn_1 = nn.BatchNorm2d(8)

        self.conv_2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.bn_2 = nn.BatchNorm2d(16)
        self.skip_conv_2 = nn.Conv2d(16, 5, 5, stride=1, padding=2)

        self.conv_3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.bn_3 = nn.BatchNorm2d(32)
        self.skip_conv_3 = nn.Conv2d(32, 5, 5, stride=1, padding=2)

        self.conv_4 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.bn_4 = nn.BatchNorm2d(64)
        self.skip_conv_4 = nn.Conv2d(64, 5, 5, stride=1, padding=2)

        self.conv_5 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.bn_5 = nn.BatchNorm2d(128)
        self.skip_conv_5 = nn.Conv2d(128, 5, 5, stride=1, padding=2)

        self.conv_6 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.bn_6 = nn.BatchNorm2d(256)

        self.upconv_5 = nn.ConvTranspose2d(256, 123, 4, stride=2, padding=1)
        self.upbn_5 = nn.BatchNorm2d(128)

        self.upconv_4 = nn.ConvTranspose2d(128, 59, 4, stride=2, padding=1)
        self.upbn_4 = nn.BatchNorm2d(64)

        self.upconv_3 = nn.ConvTranspose2d(64, 27, 4, stride=2, padding=1)
        self.upbn_3 = nn.BatchNorm2d(32)

        self.upconv_2 = nn.ConvTranspose2d(32, 11, 4, stride=2, padding=1)
        self.upbn_2 = nn.BatchNorm2d(16)

        self.upconv_1 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
        self.upbn_1 = nn.BatchNorm2d(8)

        self.out_conv = nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1)
        self.out_bn = nn.BatchNorm2d(3)

    def forward(self, z):
        x = self.leaky_relu(self.bn_1(self.conv_1(z)))
        
        x = self.leaky_relu(self.bn_2(self.conv_2(x)))
        skip_2 = self.skip_conv_2(x)

        x = self.leaky_relu(self.bn_3(self.conv_3(x)))
        skip_3 = self.skip_conv_3(x)

        x = self.leaky_relu(self.bn_4(self.conv_4(x)))
        skip_4 = self.skip_conv_4(x)

        x = self.leaky_relu(self.bn_5(self.conv_5(x)))
        skip_5 = self.skip_conv_5(x)

        x = self.leaky_relu(self.bn_6(self.conv_6(x)))

        x = torch.cat([self.upconv_5(x), skip_5], dim=1)
        x = self.leaky_relu(self.upbn_5(x))

        x = torch.cat([self.upconv_4(x), skip_4], dim=1)
        x = self.leaky_relu(self.upbn_4(x))

        x = torch.cat([self.upconv_3(x), skip_3], dim=1)
        x = self.leaky_relu(self.upbn_3(x))

        x = torch.cat([self.upconv_2(x), skip_2], dim=1)
        x = self.leaky_relu(self.upbn_2(x))

        x = self.leaky_relu(self.upbn_1(self.upconv_1(x)))

        out = nn.Sigmoid()(self.out_bn(self.out_conv(x)))

        return out