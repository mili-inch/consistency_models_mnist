import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim=256, cond_embed_dim=256):
        super().__init__()
        self.gn1 = nn.GroupNorm(1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_embed = nn.Linear(time_embed_dim, out_channels)
        self.cond_embed = nn.Linear(cond_embed_dim, out_channels)
        self.silu = nn.SiLU()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, time_embed, cond_embed):
        r = self.gn1(x)
        r = self.silu(r)
        r = self.conv1(r)
        time_embed = self.silu(time_embed)
        time_embed = self.time_embed(time_embed)
        time_embed = time_embed.unsqueeze(-1).unsqueeze(-1)
        cond_embed = self.silu(cond_embed)
        cond_embed = self.cond_embed(cond_embed)
        cond_embed = cond_embed.unsqueeze(-1).unsqueeze(-1)
        r = r + time_embed + cond_embed
        r = self.gn2(r)
        r = self.silu(r)
        r = self.conv2(r)
        if hasattr(self, 'skip'):
            x = self.skip(x)
        x = x + r
        return x

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, channels, cond_embed_dim=256, time_embed_dim=256):
        super().__init__()
        self.in_conv = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.down1_1 = ResBlock(64, 64, time_embed_dim, cond_embed_dim)
        self.down1_2 = ResBlock(64, 64, time_embed_dim, cond_embed_dim)
        self.down1_3 = ResBlock(64, 64, time_embed_dim, cond_embed_dim)

        self.down2_1 = Downsample(64)
        self.down2_2 = ResBlock(64, 128, time_embed_dim, cond_embed_dim)
        self.down2_3 = ResBlock(128, 128, time_embed_dim, cond_embed_dim)
        self.down2_4 = ResBlock(128, 128, time_embed_dim, cond_embed_dim)
        self.down2_5 = ResBlock(128, 128, time_embed_dim, cond_embed_dim)

        self.mid_1 = Downsample(128)
        self.mid_2 = ResBlock(128, 256, time_embed_dim, cond_embed_dim)
        self.mid_3 = ResBlock(256, 256, time_embed_dim, cond_embed_dim)
        self.mid_4 = ResBlock(256, 256, time_embed_dim, cond_embed_dim)
        self.mid_5 = ResBlock(256, 256, time_embed_dim, cond_embed_dim)
        self.mid_6 = ResBlock(256, 256, time_embed_dim, cond_embed_dim)
        self.mid_7 = ResBlock(256, 256, time_embed_dim, cond_embed_dim)
        self.mid_8 = ResBlock(256, 256, time_embed_dim, cond_embed_dim)
        self.mid_9 = ResBlock(256, 256, time_embed_dim, cond_embed_dim)
        self.mid_10 = ResBlock(256, 256, time_embed_dim, cond_embed_dim)
        self.mid_11 = ResBlock(256, 256, time_embed_dim, cond_embed_dim)
        self.mid_12 = Upsample(256)
        self.mid_13 = ResBlock(256, 128, time_embed_dim, cond_embed_dim)

        self.up1_1 = ResBlock(256, 128, time_embed_dim, cond_embed_dim)
        self.up1_2 = ResBlock(128, 128, time_embed_dim, cond_embed_dim)
        self.up1_3 = ResBlock(128, 128, time_embed_dim, cond_embed_dim)
        self.up1_4 = ResBlock(128, 128, time_embed_dim, cond_embed_dim)
        self.up1_5 = Upsample(128)
        self.up1_6 = ResBlock(128, 64, time_embed_dim, cond_embed_dim)

        self.up2_1 = ResBlock(128, 64, time_embed_dim, cond_embed_dim)
        self.up2_2 = ResBlock(64, 64, time_embed_dim, cond_embed_dim)
        self.up2_3 = ResBlock(64, 64, time_embed_dim, cond_embed_dim)
        self.up2_4 = ResBlock(64, 64, time_embed_dim, cond_embed_dim)

        self.out_gn = nn.GroupNorm(1, 64)
        self.silu = nn.SiLU()
        self.out_conv = nn.Conv2d(64, channels, kernel_size=3, padding=1)

    def forward(self, x, time_embed, cond_embed):
        x = self.in_conv(x)
        x = self.down1_1(x, time_embed, cond_embed)
        x = self.down1_2(x, time_embed, cond_embed)
        x = self.down1_3(x, time_embed, cond_embed)
        x_1 = self.down2_1(x)
        x_1 = self.down2_2(x_1, time_embed, cond_embed)
        x_1 = self.down2_3(x_1, time_embed, cond_embed)
        x_1 = self.down2_4(x_1, time_embed, cond_embed)
        x_1 = self.down2_5(x_1, time_embed, cond_embed)
        x_2 = self.mid_1(x_1)
        x_2 = self.mid_2(x_2, time_embed, cond_embed)
        x_2 = self.mid_3(x_2, time_embed, cond_embed)
        x_2 = self.mid_4(x_2, time_embed, cond_embed)
        x_2 = self.mid_5(x_2, time_embed, cond_embed)
        x_2 = self.mid_6(x_2, time_embed, cond_embed)
        x_2 = self.mid_7(x_2, time_embed, cond_embed)
        x_2 = self.mid_8(x_2, time_embed, cond_embed)
        x_2 = self.mid_9(x_2, time_embed, cond_embed)
        x_2 = self.mid_10(x_2, time_embed, cond_embed)
        x_2 = self.mid_11(x_2, time_embed, cond_embed)
        x_2 = self.mid_12(x_2)
        x_2 = self.mid_13(x_2, time_embed, cond_embed)
        x_1 = th.cat([x_1, x_2], dim=1)
        x_1 = self.up1_1(x_1, time_embed, cond_embed)
        x_1 = self.up1_2(x_1, time_embed, cond_embed)
        x_1 = self.up1_3(x_1, time_embed, cond_embed)
        x_1 = self.up1_4(x_1, time_embed, cond_embed)
        x_1 = self.up1_5(x_1)
        x_1 = self.up1_6(x_1, time_embed, cond_embed)
        x = th.cat([x, x_1], dim=1)
        x = self.up2_1(x, time_embed, cond_embed)
        x = self.up2_2(x, time_embed, cond_embed)
        x = self.up2_3(x, time_embed, cond_embed)
        x = self.up2_4(x, time_embed, cond_embed)
        x = self.out_gn(x)
        x = self.silu(x)
        x = self.out_conv(x)
        return x