

import torch
import torch.nn as nn
import torch.nn.functional as F



def activation(x):
    return F.leaky_relu(x, negative_slope=0.01)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.norm = nn.InstanceNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = activation(x)
        return x



class DeviationNet(nn.Module):
    """
    CNN that predicts raw deviations (u_raw, v_raw)


    - outputs direction only (no constraints)
    """

    def __init__(self, n_channels, hidden=32, n_layers=2):
        super().__init__()

        self.n_channels = n_channels

        in_ch = 5 * n_channels
        out_ch = 2 * n_channels  # u + v

        layers = []

        layers.append(ConvBlock(in_ch, hidden))

   
        for _ in range(n_layers - 1):
            layers.append(ConvBlock(hidden, hidden))

        self.body = nn.Sequential(*layers)
        self.final = nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1)


    @staticmethod
    def pack(blocks):
        return torch.cat(blocks, dim=1)

    @staticmethod
    def unpack(tensor, shapes):
        out = []
        c = 0
        for s in shapes:
            ch = s[1]
            out.append(tensor[:, c:c + ch])
            c += ch
        return out



    def forward(
        self,
        x_blocks,
        p_blocks,
        y_blocks,
        z_blocks,
        noisy,
        shapes
    ):
        """
        Inputs:
            blocks: lists of tensors [B, C, H, W]
            noisy: [B,1,H,W]

        Output:
            u_raw, v_raw (lists of blocks)
        """


        noisy_rep = noisy.repeat(1, self.n_channels, 1, 1)
    
        inp = torch.cat([
            self.pack(x_blocks),
            self.pack(p_blocks),
            self.pack(y_blocks),
            self.pack(z_blocks),
            noisy_rep
        ], dim=1)


        h = self.body(inp)
        out = self.final(h)

   
        u_raw = self.unpack(out[:, :self.n_channels], shapes)
        v_raw = self.unpack(out[:, self.n_channels:], shapes)


        u_raw = [torch.nan_to_num(u) for u in u_raw]
        v_raw = [torch.nan_to_num(v) for v in v_raw]

        return u_raw, v_raw