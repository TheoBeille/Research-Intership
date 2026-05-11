

import torch
import torch.nn as nn
import torch.nn.functional as F


from Algo_setup_tomo import Params
params=Params()
size=params.size

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

        in_ch = 7  * n_channels
        out_ch = 2 * n_channels  # u + v
        self.input_norm = nn.InstanceNorm2d(in_ch)
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
        shapes,
        x_blocks,
        p_blocks,
        y_blocks,
        z_blocks,
        u_prev,
        v_prev,
        Cy,

    ):
        """
        Inputs:
            blocks: lists of tensors [B, C, H, W]
            initial_state: [B,1,H,W]

        Output:
            u_raw, v_raw (lists of blocks)
        """


    

        inputs = []


        inputs.append(self.pack(x_blocks[:2]))

        inputs.append(self.pack(p_blocks[:2]))

        inputs.append(self.pack(y_blocks[:2]))

        inputs.append(self.pack(z_blocks[:2]))

        inputs.append(self.pack(u_prev[:2]))

        inputs.append(self.pack(v_prev[:2]))
        inputs.append(self.pack(Cy[:2]))
        inp = torch.cat(inputs, dim=1)
        
        

        inp = self.input_norm(inp)


        h = self.body(inp)
        out = self.final(h)
        B = out.shape[0]
        idx = 0

        u_learned = []
        v_learned = []


        for i in range(2):
            ch = shapes[i][1]
            u_learned.append(torch.nan_to_num(out[:, idx:idx+ch]))
            idx += ch

        for i in range(2):
            ch = shapes[i][1]
            v_learned.append(torch.nan_to_num(out[:, idx:idx+ch]))
            idx += ch

        device = out.device

        u_zeros = [
            torch.zeros(B, shapes[2][1], size, size, device=device),
            torch.zeros(B, shapes[3][1], size, size, device=device)
        ]

        v_zeros = [
            torch.zeros(B, shapes[2][1], size, size, device=device),
            torch.zeros(B, shapes[3][1], size, size, device=device)
        ]

        u_raw = u_learned + u_zeros
        v_raw = v_learned + v_zeros

        return u_raw, v_raw

        