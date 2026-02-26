
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np
import math

# input: B, C, H, W
# flow: [B, 2, H, W]
def wrap(input, flow):
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(input.device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(input.device)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid, align_corners=True)
    #output_pre = torch.nn.functional.grid_sample(input, vgrid, mode='nearest',align_corners=True)
    return output

def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=4, num_channels=in_channels, eps=1e-6, affine=True)#32

def swish(x):
    return x*torch.sigmoid(x)
    
class unfold32(nn.Module):
    """
    ps = 32,stride = 32
    B,T,C,H,W -> B*phn*pwn,T,C,ps,ps
    """
    def __init__(self,size = 32) -> None:
        super().__init__()
        self.size = size
        self.unfold = nn.Unfold(kernel_size=size, stride=size)
    def forward(self,x):
        B, T, C, H, W = x.shape
        ps =self.size
        stride = self.size
        phn = H//stride
        pwn = W//stride
        x = x.reshape(B*T,C,H,W)
        x = self.unfold(x)#->B*T,ps*ps,phn*pwn
        x= x.reshape(-1,T,ps,ps,phn,pwn)
        x = x.permute(0, 4, 5, 1, 2, 3).reshape(-1,T,C,ps,ps)
        return x
    
class fold32(nn.Module):
    """
    ps = 32,stride = 32
    B*phn*pwn,T,C,ps,ps->B,T,C,H,W
    warn:phm*ps!=H
    """
    def __init__(self,size = 32,H = 1024) -> None:
        super().__init__()
        self.H = H
        self.size = size
        self.fold=nn.Fold(output_size=(H,H),kernel_size=size,stride=size)

    def forward(self,x):
        B_,T,C,ps,ps = x.shape
        h = self.H//self.size
        x = x.reshape(-1,h,h,12,ps,ps).permute(0,3,4,5,1,2)
        x = x.reshape(-1,ps*ps,h*h)
        x = self.fold(x).reshape(-1,12,1,self.H,self.H)

        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x

class ResBlock(nn.Module):
    #swish groupnorm
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class ConvGRUCell(nn.Module):
    """A ConvGRU implementation."""
    """bt c h w -> bt out_c h w"""
    def __init__(self, in_channels, out_channels, kernel_size, out_shape = [64,128,128], mode = 0):
        super().__init__()
        self.mode = mode
        self.out_shape = out_shape
        same_padding = int((kernel_size[0]-1)/2)
        self._sn_eps = 0.0001
        if mode == 1:
            self.resize = Downsample(in_channels,out_channels)#下采样
        elif mode == 2:
            self.resize = Upsample(in_channels,out_channels)#上采样
        
        self.conv2d_x = nn.Conv2d(in_channels=out_channels, out_channels=2*out_channels,
                                  kernel_size=kernel_size, stride=1, padding=same_padding)
        self.conv2d_h = nn.Conv2d(in_channels=out_channels, out_channels=2*out_channels,
                                  kernel_size=kernel_size, stride=1, padding=same_padding)
        
        self.conv2d_h2h = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                                    kernel_size= kernel_size,stride=1,padding=same_padding)
        self.conv2d_x2h = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                                    kernel_size= kernel_size,stride=1,padding=same_padding)
    def forward(self, x, h=None):
        bs = x.shape[0]
        if h == None:
            h = torch.zeros(bs,self.out_shape[0], self.out_shape[1], self.out_shape[2]).to(x.device)
            c = torch.zeros(bs,self.out_shape[0], self.out_shape[1], self.out_shape[2]).to(x.device)
        if self.mode!=0:
            x = self.resize(x)

        x_after_conv = self.conv2d_x(x)
        h_after_conv = self.conv2d_h(h)

        xz, xr = torch.chunk(x_after_conv, 2, dim=1)
        hz, hr = torch.chunk(h_after_conv, 2, dim=1)
        
        zt = torch.sigmoid(xz+hz)#update zero gate
        rt = torch.sigmoid(xr+hr)#read gate

        x2h = self.conv2d_x2h(x)
        h2h = self.conv2d_h2h(rt * h)
        c = F.relu(x2h+h2h)
        
        out = zt * h + (1.0 - zt) * c
        new_h = out

        return new_h, out

class ConvGRU(nn.Module):
    """ConvGRU Cell wrapper to replace tf.static_rnn in TF implementation"""
    """"1layer for grucell  t b c h w -> t b c h w"""

    def __init__(self,in_channels,out_channels,kernel_size, out_shape = [64,128,128], mode = 0):
        super().__init__()
        self.cell = ConvGRUCell(in_channels, out_channels, kernel_size, out_shape, mode)

    def forward(self, x: torch.Tensor, ht=None) -> torch.Tensor:
        x = x.permute(1,0,2,3,4)
        outputs = []
        for step in range(len(x)):
            # Compute current timestep
            output, ht = self.cell(x[step], ht)
            outputs.append(output)
        # Stack outputs to return as tensor
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.permute(1,0,2,3,4)
        return outputs,ht

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x,y):
        """
        x:q
        y:k,v
        """
        h_ = x
        y_ = y
        h_ = self.norm(h_)
        y_ = self.norm(y_)
        q = self.q(h_)
        k = self.k(y_)
        v = self.v(y_)

        # compute attention
        b, c, h, w = q.shape
        b, c, h1,w1 = k.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   
        k = k.reshape(b, c, h1*w1)
        w_ = torch.bmm(q, k) 
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h1*w1)
        w_ = w_.permute(0, 2, 1) 
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_
