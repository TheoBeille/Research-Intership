import torch
import torch.nn as nn
import torch.nn.functional as F

from Algo_setuptorch import Params

params = Params()
size = params.size


# ============================================================
# ACTIVATION
# ============================================================

def activation(x):
    return F.leaky_relu(x, negative_slope=0.01)


# ============================================================
# DOUBLE CONV
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.01),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# DOWN BLOCK
# ============================================================

class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):

        x = self.pool(x)

        x = self.conv(x)

        return x


# ============================================================
# UP BLOCK
# ============================================================

class Up(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            out_ch + skip_ch,
            out_ch
        )

    def forward(self, x, skip):

        x = self.up(x)

        # padding safety
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]

        x = F.pad(
            x,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
            ]
        )

        x = torch.cat([skip, x], dim=1)

        x = self.conv(x)

        return x


# ============================================================
# MAIN U-NET
# ============================================================

class DeviationNet(nn.Module):

    def __init__(
        self,
        n_channels,
        base=32,
    ):

        super().__init__()

        self.n_channels = n_channels

        # ====================================================
        # INPUT CHANNELS
        # ====================================================

        in_ch = 7 * n_channels

        out_ch = 2 * n_channels

        # ====================================================
        # TIME EMBEDDING
        # ====================================================

        self.time_mlp = nn.Sequential(

            nn.Linear(1, base),

            nn.LeakyReLU(0.01),

            nn.Linear(base, base)
        )

        # ====================================================
        # ENCODER
        # ====================================================

        self.inc = DoubleConv(in_ch, base)

        self.down1 = Down(base, 2 * base)

        self.down2 = Down(2 * base, 4 * base)

        self.down3 = Down(4 * base, 8 * base)

        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.bottleneck = DoubleConv(
            8 * base,
            8 * base
        )

        # ====================================================
        # DECODER
        # ====================================================

        self.up1 = Up(
            8 * base,
            4 * base,
            4 * base
        )

        self.up2 = Up(
            4 * base,
            2 * base,
            2 * base
        )

        self.up3 = Up(
            2 * base,
            base,
            base
        )

        # ====================================================
        # FINAL
        # ====================================================

        self.final = nn.Conv2d(
            base,
            out_ch,
            kernel_size=1
        )

    # ========================================================
    # PACK / UNPACK
    # ========================================================

    @staticmethod
    def pack(blocks):
        return torch.cat(blocks, dim=1)

    # ========================================================
    # FORWARD
    # ========================================================

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
        t,
    ):

        # ====================================================
        # INPUTS
        # ====================================================

        inputs = [

            self.pack(x_blocks[:2]),
            self.pack(p_blocks[:2]),
            self.pack(y_blocks[:2]),
            self.pack(z_blocks[:2]),
            self.pack(u_prev[:2]),
            self.pack(v_prev[:2]),
            self.pack(Cy[:2]),
        ]

        x = torch.cat(inputs, dim=1)

        # ====================================================
        # ENCODER
        # ====================================================

        x1 = self.inc(x)

        # ----------------------------------------------------
        # TIME EMBEDDING
        # ----------------------------------------------------

        if t.dim() == 1:
            t = t.unsqueeze(1)

        t_embed = self.time_mlp(t)

        t_embed = t_embed.unsqueeze(-1).unsqueeze(-1)

        x1 = x1 + t_embed

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        # ====================================================
        # BOTTLENECK
        # ====================================================

        h = self.bottleneck(x4)

        # ====================================================
        # DECODER
        # ====================================================

        h = self.up1(h, x3)

        h = self.up2(h, x2)

        h = self.up3(h, x1)

        # ====================================================
        # OUTPUT
        # ====================================================

        out = self.final(h)

        B = out.shape[0]

        idx = 0

        u_learned = []
        v_learned = []

        # ----------------------------------------------------
        # u corrections
        # ----------------------------------------------------

        for i in range(2):

            ch = shapes[i][1]

            u_learned.append(

                torch.nan_to_num(
                    out[:, idx:idx+ch]
                )
            )

            idx += ch

        # ----------------------------------------------------
        # v corrections
        # ----------------------------------------------------

        for i in range(2):

            ch = shapes[i][1]

            v_learned.append(

                torch.nan_to_num(
                    out[:, idx:idx+ch]
                )
            )

            idx += ch

        # ====================================================
        # NO CORRECTION ON DUAL TGV BLOCKS
        # ====================================================

        device = out.device

        u_zeros = [

            torch.zeros(
                B,
                shapes[2][1],
                size,
                size,
                device=device
            ),

            torch.zeros(
                B,
                shapes[3][1],
                size,
                size,
                device=device
            )
        ]

        v_zeros = [

            torch.zeros(
                B,
                shapes[2][1],
                size,
                size,
                device=device
            ),

            torch.zeros(
                B,
                shapes[3][1],
                size,
                size,
                device=device
            )
        ]

        u_raw = u_learned + u_zeros

        v_raw = v_learned + v_zeros

        return u_raw, v_raw