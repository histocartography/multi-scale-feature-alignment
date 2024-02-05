import numpy as np
import torch
from torch import nn

from source.utils.layers import (
    CBAM,
    EQLinear,
    EQConv2d,
    StyleEQConv2dWithBias,
    MinibatchStdLayer,
)


class Generator(nn.Module):

    def __init__(self, latent_dim, n_cls, device):
        super(Generator, self).__init__()
        # In position 0, dummy placeholder
        self.input_res = [1, 4, 4, 8, 16, 32, 64, 128, 128]
        print(f"Gen n_cls: {n_cls}")
        self.latent_dim = latent_dim
        # dimension of linearly embedded onehot
        onehot_embed_dim = 256
        # activation used whenever non-linear
        self.act = nn.LeakyReLU(0.2)
        # bilinear upsampling layer
        self.biliUp = nn.UpsamplingBilinear2d(scale_factor=2)

        # embed onehot
        self.onehot_embed = EQLinear(n_cls, onehot_embed_dim, bias=True)
        # map latent + onehot to w
        self.w1 = EQLinear(
            latent_dim + onehot_embed_dim, latent_dim, bias=True, lr_mul=0.01
        )
        self.w2 = EQLinear(latent_dim, latent_dim, bias=True, lr_mul=0.01)

        # no. of in channels foreach layer
        in_nchans = [None, 512, 512, 256, 256, 128, 128, 64, 64, 64]
        base_res = 4
        base_nchan = 512
        # learned constant
        self.const = nn.Parameter(torch.ones((1, base_nchan, base_res, base_res)))
        # conv layers, styles, and noise scales
        for i in range(1, 9):
            in_nchan = in_nchans[i]
            out_nchan = in_nchans[i + 1]
            conv = StyleEQConv2dWithBias(
                in_nchan,
                out_nchan,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                device=device,
            )
            setattr(self, "conv{}".format(i), conv)
        # output layer (no noise)
        self.out_layer = StyleEQConv2dWithBias(
            in_nchans[-1], 3, kernel_size=3, stride=1, padding=1, bias=True, noise=False
        )

        self.to(device)

    def requires_grad(self, value: bool):
        for param in self.parameters():
            param.requires_grad = value

    def forward(self, z, one_hot, noise=None):

        # embed (linearly) and concat
        one_hot = self.onehot_embed(one_hot)
        x = torch.cat([z, one_hot], dim=1)

        # map to w
        w = self.act(self.w1(x))
        w = self.act(self.w2(w))

        bs = x.size()[0]
        # broadcast learned constant along batch dim
        x = self.const.expand([bs, -1, -1, -1])
        for i in range(1, 9):
            style_conv = getattr(self, "conv{}".format(i))
            n = noise[i] if noise is not None else None
            x = style_conv((x, w), n=n)
            x = self.act(x)
            # [None, 4, 4, 8, 16, 32, 64, 128, 128]
            if 1 < i < 7:
                x = self.biliUp(x)
            # 512x4x4 -> 512x4x4 1
            # 512x4x4 -> 256x4x4 -> 256x8x8 2
            # 256x8x8 -> 256x8x8 -> 256x16x16 3
            # 256x16x16 -> 128x16x16 -> 128x32x32 4
            # 128x32x32 -> 128x32x32 -> 128x64x64 5
            # 128x64x64 -> 64x64x64 -> 64x128x128 6
            # 64x128x128 -> 64x128x128 -> 64x256x256 7
            # 64x128x128 -> 64x128x128 8
            # 64x128x128 -> 3x128x128 output

        # linear output
        return self.out_layer((x, w))


class Prologue(nn.Module):
    def __init__(
        self, stage, in_n_chan, out_nchan=512, residual=False, attention=False
    ):
        super().__init__()
        # stage in (0, 5, 6)
        self.stage = stage
        self.residual = residual
        self.attention = attention
        # cheap downsampling
        self.biliDown = nn.AvgPool2d(2, stride=2)
        # ceiled downsampling
        self.cbiliDown = nn.AvgPool2d(2, stride=2, ceil_mode=True)
        # Activation function
        self.act = nn.LeakyReLU(0.2)
        self.pool_one = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()

        if stage == 0:
            base = 32
            # 3x256x256 -> 32x256x256
            self.conv1 = EQConv2d(
                in_n_chan, base, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 32x128x128 -> 64x128x128
            self.conv2 = EQConv2d(
                base, base * 2, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 64x64x64-> 128x64x64
            self.conv3 = EQConv2d(
                base * 2, base * 4, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 128x32x32 -> 256x32x32
            self.conv4 = EQConv2d(
                base * 4, base * 8, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 256x16x16 -> 512x16x16
            self.conv5 = EQConv2d(
                base * 8, base * 16, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 512x8x8 -> 512x8x8
            self.conv6 = EQConv2d(
                base * 16, base * 16, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 512x4x4 -> 512x4x4
            self.conv7 = EQConv2d(
                base * 16, out_nchan, kernel_size=3, stride=1, padding=1, bias=True
            )
        else:
            if stage <= 5:
                base = 128
                # ?x32x32 -> 128x32x32
                self.conv5_1 = EQConv2d(
                    in_n_chan, base, kernel_size=3, stride=1, padding=1, bias=True
                )
                # 128x32x32 -> ?x32x32
                self.conv5_2 = EQConv2d(
                    base, in_n_chan, kernel_size=3, stride=1, padding=1, bias=True
                )
                # 2?x16x16 -> ?x16x16
                self.ingest6 = EQConv2d(
                    in_n_chan * 3,
                    in_n_chan * 2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
                self.cbam_5 = CBAM(in_n_chan) if self.attention else None
                in_n_chan *= 2

            if stage <= 6:
                base = 256
                # 2?x16x16 -> 256x16x16
                self.conv6_1 = EQConv2d(
                    in_n_chan, base, kernel_size=3, stride=1, padding=1, bias=True
                )
                # 256x16x16 -> 2?x16x16
                self.conv6_2 = EQConv2d(
                    base, in_n_chan, kernel_size=3, stride=1, padding=1, bias=True
                )
                # 4?x16x16 -> 2?x16x16
                self.ingest7 = EQConv2d(
                    in_n_chan * 3,
                    in_n_chan * 2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
                self.cbam_6 = CBAM(in_n_chan) if self.attention else None
                in_n_chan *= 2

            if stage <= 7:
                base = 512 if stage == 7 else 256
                # 4?x8x8 -> 512x4x4
                self.conv1 = EQConv2d(
                    in_n_chan, base, kernel_size=3, stride=1, padding=1, bias=True
                )
                self.conv2 = EQConv2d(
                    base, out_nchan, kernel_size=3, stride=1, padding=1, bias=True
                )

            else:
                raise ValueError(
                    "Discriminator not implemented for stage {}".format(stage)
                )

    def forward(self, x):
        if self.stage == 0:
            # 3x256x256
            x = self.act(self.conv1(x))
            # 32x128x128
            x = self.act(self.conv2(x))
            x = self.biliDown(x)
            # 64x64x64
            x = self.act(self.conv3(x))
            x = self.biliDown(x)
            # 128x32x32
            x = self.act(self.conv4(x))
            x = self.biliDown(x)
            # 256x16x16
            x = self.act(self.conv5(x))
            x = self.biliDown(x)
            # 512x8x8
            x = self.act(self.conv6(x))
            x = self.biliDown(x)
            # 512x4x4
            x = self.act(self.conv7(x))
            return self.biliDown(x)

        else:
            if self.stage <= 5:
                x_5, x_6, x_7 = x
                # 128x32x32 -> 128x32x32
                x_5c = self.act(self.conv5_1(x_5))
                x_5c = self.act(self.conv5_2(x_5c))
                # 128x32x32 -> 128x16x16
                x_5c = self.biliDown(x_5c)
                # (residual) + (128x32x32 -> 128x16x16)
                if self.attention:
                    x_5c = self.cbam_5(x_5c)
                if self.residual:
                    x_5c = x_5c + self.biliDown(x_5)
                # (256 + 128)x16x16 -> 256x16x16
                x_6 = self.ingest6(torch.cat((x_6, x_5c), dim=1))
                # if not residual

            if self.stage <= 6:
                if self.stage == 6:
                    x_6, x_7 = x
                # 256x16x16 -> 256x16x16
                x_6c = self.act(self.conv6_1(x_6))
                x_6c = self.act(self.conv6_2(x_6c))
                # 256x16x16 -> 256x8x8
                x_6c = self.biliDown(x_6c)
                # (residual) + (256x16x16 -> 256x8x8)
                if self.attention:
                    x_6c = self.cbam_6(x_6c)
                if self.residual:
                    x_6c = x_6c + self.biliDown(x_6)
                # (512 + 256)x8x8 -> 512x8x8
                x_7 = self.ingest7(torch.cat((x_7, x_6c), dim=1))

            if self.stage <= 7:
                if self.stage == 7:
                    x_7 = x[0]
                # 256x8x8 -> 256|512x8x8
                x_7c = self.act(self.conv1(x_7))
                x_7c = self.act(self.conv2(x_7c))
                # 512x4x4 -> 512x2x2
                x_7c = self.biliDown(x_7c)

            return x_7c


class Epilogue(nn.Module):
    def __init__(
        self,
        stage,
        in_channels,
        res,
        n_cls=0,
        mbdis_group_size=32,
        mbdis_n_chan=0,
        cmap_dim=128,
    ):
        super().__init__()
        self.cmap_dim = cmap_dim
        self.stage = stage
        self.mbdis = None
        self.act = nn.LeakyReLU(0.2)

        # handcrafted mbdis features
        self.mbdis = (
            MinibatchStdLayer(mbdis_group_size, mbdis_n_chan)
            if mbdis_n_chan > 0
            else None
        )
        # last conv layer incorporates mbdis features
        self.conv = EQConv2d(
            in_channels + mbdis_n_chan, in_channels, kernel_size=3, padding=1
        )
        # dense layer instead of further downsampling
        self.dense = EQLinear(in_channels * (res**2), in_channels, bias=True)
        # output layer (maps to cmap_dim outputs instead of single logit)
        self.logits = EQLinear(in_channels, cmap_dim, bias=True)
        # projection layer for condition label (affine)
        self.onehot_project = EQLinear(n_cls, cmap_dim, bias=False)

    def forward(self, x, c):
        if self.mbdis is not None:
            x = self.mbdis(x)
        x = self.act(self.conv(x))
        # dense layer that does 'downsampling'
        x = self.act(self.dense(x.flatten(1)))
        # project condition
        c_proj = self.onehot_project(c)
        logits_t = self.logits(x)
        out_t = (logits_t * c_proj).sum(dim=1, keepdim=True) * (
            1 / np.sqrt(self.cmap_dim)
        )  # TODO: verify

        return out_t


class Discriminator(nn.Module):

    def __init__(
        self, stage, n_cls, device, mbdis=True, attention=False, residual=False
    ):
        super(Discriminator, self).__init__()
        print(f"Disc n_cls: {n_cls}")
        # prologue depends on replay stage
        assert stage in (0, 5, 6, 7), f"Discriminator not implemented for stage {stage}"
        self.stage = stage

        # interface between prologue and epilogue
        epilogue_in_res = 2
        epilogue_in_nchan = 512

        if stage == 7:
            prologue_in_chan = 512
        elif stage == 6:
            prologue_in_chan = 256
        elif stage == 5:
            prologue_in_chan = 128
        else:
            prologue_in_chan = 3

        self.prologue = Prologue(
            stage,
            in_n_chan=prologue_in_chan,
            out_nchan=epilogue_in_nchan,
            attention=attention,
            residual=residual,
        )
        # epilogue, with optional mini-batch discrimination layer
        mbdis_n_chan = 1 if mbdis else 0
        self.epilogue = Epilogue(
            stage,
            epilogue_in_nchan,
            n_cls=n_cls,
            res=epilogue_in_res,
            mbdis_n_chan=mbdis_n_chan,
        )
        self.to(device)

    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def forward(self, x, c):
        x = self.prologue(x)
        x = self.epilogue(x, c)
        return x


class G_D(nn.Module):

    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D
