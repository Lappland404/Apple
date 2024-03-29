from unet_parts import *

class UNet_model(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = False):
        super(UNet_model, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.dconv = DoubleConv(n_channels, 64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,512)
        self.up1 = Up(1024,256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.output = outConv(64,n_classes)


    def forward(self, x):
        out1 = self.dconv(x)
        out2 = self.down1(out1)
        out3 = self.down2(out2)
        out4 = self.down3(out3)
        out5 = self.down4(out4)
        x = self.up1(out5, out4)
        x = self.up2(x, out3)
        x = self.up3(x, out2)
        x = self.up4(x, out1)
        pre = self.output(x)
        return pre

if __name__ == '__main__':
    net = UNet_model(n_channels=3, n_classes=1)
    print(net)
