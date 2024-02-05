import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch import Tensor


# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
    # Apply scale and shift--if gain and bias are provided, fuse them here
    # Prepare scale
    scale = torch.rsqrt(var + eps)
    # If a gain is provided, use it
    if gain is not None:
        scale = scale * gain
    # Prepare shift
    shift = mean * scale
    # If bias is provided, use it
    if bias is not None:
        shift = shift - bias
    return x * scale - shift


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
    # Cast x to float32 if necessary
    float_x = x.float()
    # Calculate expected value of x (m) and expected value of x**2 (m2)
    # Mean of x
    m = torch.mean(float_x, [0, 2, 3], keepdim=True)
    # Mean of x squared
    m2 = torch.mean(float_x**2, [0, 2, 3], keepdim=True)
    # Calculate variance as mean of squared minus mean squared.
    var = m2 - m**2
    # Cast back to float 16 if necessary
    var = var.type(x.type())
    m = m.type(x.type())
    # Return mean and variance for updating stored mean/var if requested
    if return_mean_var:
        return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
    else:
        return fused_bn(x, m, var, gain, bias, eps)


# My batchnorm, supports standing stats
class myBN(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super(myBN, self).__init__()
        # momentum for updating running stats
        self.momentum = momentum
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Register buffers
        self.register_buffer("stored_mean", torch.zeros(num_channels))
        self.register_buffer("stored_var", torch.ones(num_channels))
        self.register_buffer("accumulation_counter", torch.zeros(1))
        # Accumulate running means and vars
        self.accumulate_standing = False

    # reset standing stats
    def reset_stats(self):
        self.stored_mean[:] = 0
        self.stored_var[:] = 0
        self.accumulation_counter[:] = 0

    def forward(self, x, gain, bias):
        if self.training:
            out, mean, var = manual_bn(
                x, gain, bias, return_mean_var=True, eps=self.eps
            )
            # If accumulating standing stats, increment them
            if self.accumulate_standing:
                self.stored_mean[:] = self.stored_mean + mean.data
                self.stored_var[:] = self.stored_var + var.data
                self.accumulation_counter += 1.0
            # If not accumulating standing stats, take running averages
            else:
                self.stored_mean[:] = (
                    self.stored_mean * (1 - self.momentum) + mean * self.momentum
                )
                self.stored_var[:] = (
                    self.stored_var * (1 - self.momentum) + var * self.momentum
                )
            return out
        # If not in training mode, use the stored statistics
        else:
            mean = self.stored_mean.view(1, -1, 1, 1)
            var = self.stored_var.view(1, -1, 1, 1)
            # If using standing stats, divide them by the accumulation counter
            if self.accumulate_standing:
                mean = mean / self.accumulation_counter
                var = var / self.accumulation_counter
            return fused_bn(x, mean, var, gain, bias, self.eps)


# Simple function to handle groupnorm norm stylization
def groupnorm(x, norm_style):
    # If number of channels specified in norm_style:
    if "ch" in norm_style:
        ch = int(norm_style.split("_")[-1])
        groups = max(int(x.shape[1]) // ch, 1)
    # If number of groups specified in norm style
    elif "grp" in norm_style:
        groups = int(norm_style.split("_")[-1])
    # If neither, default to groups = 16
    else:
        groups = 16
    return F.group_norm(x, groups)


# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable).
class ccbn(nn.Module):
    def __init__(
        self,
        output_size,
        input_size,
        which_linear,
        eps=1e-5,
        momentum=0.1,
        cross_replica=False,
        mybn=False,
        norm_style="bn",
    ):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # Norm style?
        self.norm_style = norm_style

        if self.mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        elif self.norm_style in ["bn", "in"]:
            self.register_buffer("stored_mean", torch.zeros(output_size))
            self.register_buffer("stored_var", torch.ones(output_size))

    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        # If using my batchnorm
        if self.mybn or self.cross_replica:
            return self.bn(x, gain=gain, bias=bias)
        # else:
        else:
            if self.norm_style == "bn":
                out = F.batch_norm(
                    x,
                    self.stored_mean,
                    self.stored_var,
                    None,
                    None,
                    self.training,
                    0.1,
                    self.eps,
                )
            elif self.norm_style == "in":
                out = F.instance_norm(
                    x,
                    self.stored_mean,
                    self.stored_var,
                    None,
                    None,
                    self.training,
                    0.1,
                    self.eps,
                )
            elif self.norm_style == "gn":
                out = groupnorm(x, self.normstyle)
            elif self.norm_style == "nonorm":
                out = x
            return out * gain + bias

    def extra_repr(self):
        s = "out: {output_size}, in: {input_size},"
        s += " cross_replica={cross_replica}"
        return s.format(**self.__dict__)


# Normal, non-class-conditional BN
class bn(nn.Module):
    def __init__(
        self, output_size, eps=1e-5, momentum=0.1, cross_replica=False, mybn=False
    ):
        super(bn, self).__init__()
        self.output_size = output_size
        # Prepare gain and bias layers
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn

        if mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        # Register buffers if neither of the above
        else:
            self.register_buffer("stored_mean", torch.zeros(output_size))
            self.register_buffer("stored_var", torch.ones(output_size))

    def forward(self, x, y=None):
        if self.cross_replica or self.mybn:
            gain = self.gain.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            return self.bn(x, gain=gain, bias=bias)
        else:
            return F.batch_norm(
                x,
                self.stored_mean,
                self.stored_var,
                self.gain,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )


class EQConv2d(Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        # initialize weights from normal distribution
        nn.init.normal_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)

        # equalized lr: scale for weights
        fan_in = np.prod(self.kernel_size) * self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor) -> Tensor:
        return torch.conv2d(
            input=x,
            weight=self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class EQLinear(Linear):
    def __init__(
        self, in_features, out_features, bias=True, bias_init=0.0, lr_mul=1.0
    ) -> None:
        super().__init__(in_features, out_features, bias)

        # initialize weights from normal distribution
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / lr_mul)

        # initialize bias
        self.bias = (
            nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )

        # equalized lr: scale for weights
        fan_in = self.in_features
        self.weight_scale = lr_mul / np.sqrt(fan_in)
        self.bias_scale = lr_mul

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight * self.weight_scale
        if self.bias is not None and self.bias_scale != 1:
            b = self.bias * self.bias_scale
        else:
            b = self.bias
        return F.linear(x, w, b)


class StyleEQConv2dWithBias(EQConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        wdim=512,
        stylemod=True,
        noise=True,
        device=None,
    ) -> None:
        super(StyleEQConv2dWithBias, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.device = device
        self.out_channels = out_channels
        # No bias, scale only
        self.y1 = EQLinear(wdim, in_channels, bias_init=1.0) if stylemod else None
        # Bias
        self.y2 = EQLinear(wdim, in_channels, bias_init=0.0) if stylemod else None
        # Single noise scalar
        self.noise_scale = (
            nn.Parameter(torch.zeros([1, out_channels, 1, 1])) if noise else None
        )

    def forward(self, x: tuple, n=None) -> Tensor:
        x, w = x
        bs, nchan, res = x.size()[:3]
        # Style modulation
        if self.y1 is not None and self.y2 is not None:
            y1 = self.y1(w)
            y2 = self.y2(w)
            y1 = y1.reshape(-1, nchan, 1, 1)
            y2 = y2.reshape(-1, nchan, 1, 1)
            x = x * y1 + y2
        # Convolution
        x = super(StyleEQConv2dWithBias, self).forward(x)
        # Add noise
        if self.noise_scale is not None:
            n = torch.randn((bs, 1, res, res), device=self.device) if n is None else n
            x += self.noise_scale * n

        return x


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            EQLinear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            EQLinear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        spatial_attention=True,
    ):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.spatial_attention = spatial_attention
        if spatial_attention:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if self.spatial_attention:
            x_out = self.SpatialGate(x_out)
        return x_out


class MinibatchStdLayer(nn.Module):
    def __init__(self, group_size, n_chan=1):
        super().__init__()
        self.group_size = group_size
        self.n_chan = n_chan

    def forward(self, x):
        N, C, H, W = x.shape
        G = N
        if self.group_size is not None:
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)).item()
        F = self.n_chan
        c = C // F

        # case of multi mini bathces with remainder
        store_lst = []
        remainder = N % G
        mini_bs = torch.div(N, G, rounding_mode="floor")
        if remainder and mini_bs:
            # split minibatch in n groups of size G, split channels in F groups of size c
            store_lst.append(x[:-remainder].reshape(G, mini_bs, F, c, H, W))
            store_lst.append(x[-remainder:].reshape(remainder, 1, F, c, H, W))
        # case of a "pure" remainder, no mini batches (batch is too small)
        elif remainder and not mini_bs:
            store_lst.append(x.reshape(remainder, 1, F, c, H, W))
        # case of no remainder but long batch -> ideal scenario
        elif not remainder and mini_bs:
            store_lst.append(x.reshape(G, mini_bs, F, c, H, W))

        for i, y in enumerate(store_lst):
            # shift center (per group) to zero
            y = y - y.mean(dim=0)
            # variance per group
            y = y.square().mean(dim=0)
            # stddev
            y = (y + 1e-8).sqrt()
            # average over channels and pixels
            y = y.mean(dim=[2, 3, 4])
            # reshape and tile
            y = y.reshape(-1, F, 1, 1)
            rep = G if not (remainder and i == len(store_lst) - 1) else remainder
            y = y.repeat(rep, 1, H, W)
            store_lst[i] = y
        # add to input as 'handcrafted feature channels'
        x = torch.cat([x, torch.cat(store_lst, dim=0)], dim=1)
        return x
