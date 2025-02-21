from unet.unet_parts import *
from torch.utils.checkpoint import checkpoint

class UNET(nn.Module):
    def __init__(self, n_channels, n_classes) -> None:
        super(UNET, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inp = DoubleConv(n_channels, 64)
        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        self.down3 = DownLayer(256, 512)
        self.down4 = DownLayer(512, 1024)

        self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(512, 256)
        self.up3 = UpLayer(256, 128)
        self.up4 = UpLayer(128, 64)
        self.out = OutConv(64, n_classes)

    def contracting_path(self, X):
        out1 = self.down1(X) # 128 x 280 x 280
        out2 = self.down2(out1) # 256 x 136 x 136
        out3 = self.down3(out2) # 512 x 64 x 64
        out4 = self.down4(out3) # 1024 x 28 x 28
        return out1, out2, out3, out4
    
    def expansive_path(self, c4, c3, c2, c1, c0):
        out = self.up1(c4, c3) # 512 x 52 x 52
        out = self.up2(out, c2) # 256 x 100 x 100
        out = self.up3(out, c1) # 128 x 196 x 196
        out = self.up4(out, c0) # 64 x 388 x 388
        return out
    
    def forward(self, X):
        c0 = self.inp(X) # 64 x 568 x 568
        c1, c2, c3, c4 = self.contracting_path(c0)
        e = self.expansive_path(c4, c3, c2, c1, c0)
        logits = self.out(e) # 2 x 388 x 388
        return logits
    
    def use_checkpointing(self):
      self.inp = torch.utils.checkpoint.checkpoint(self.inp)
      self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
      self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
      self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
      self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
      self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
      self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
      self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
      self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
      self.out = torch.utils.checkpoint.checkpoint(self.out)