import torch
from torch import nn
import math
from .modules_diff import (
    Residual,
    SinusoidalPosEmb,
    Upsample,
    Downsample,
    Upsample3D,
    Downsample3D,
    PreNorm,
    Block,
    Block3D,
    LinearAttention,
    get_backbone,
    ConditionalEmbedding
)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class controlnet(nn.Module):
    def __init__(
        self,
        dim=64,
        context_dim_factor=1,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        backbone="resnet",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.context_dim_factor = context_dim_factor

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            class_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
            )
            self.class_mlp = ConditionalEmbedding(num_labels=4,dim=dim)
        else:
            time_dim = None
            self.time_mlp = None

        self.zero_convs = nn.ModuleList([self.make_zero_conv(self.channels)])
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out, time_dim)),
                        get_backbone(backbone, (dim_out + int(dim_out * self.context_dim_factor), dim_out, time_dim)),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),#ATT(norm(x))+x
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            self.zero_convs.append(self.make_zero_conv(dim_out))

        mid_dim = dims[-1]
        self.mid_block1 = get_backbone(backbone, (mid_dim, mid_dim, time_dim))
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = get_backbone(backbone, (mid_dim, mid_dim, time_dim))
        self.zero_convs.append(self.make_zero_conv(mid_dim))

    def make_zero_conv(self, channels):
        return zero_module(nn.Conv2d(channels, channels, 1, padding=0))
    def init_state():
        pass

    def forward(self, x_t, control_c, time=None, c_emb=None, context=None):
        t = self.time_mlp(time)
        c_emb = self.class_mlp(c_emb)
        x =x_t + self.zero_convs[0](control_c)
        outs = []
        for idx, (backbone, backbone2, attn, downsample) in enumerate(self.downs):
            x = backbone(x, t, c_emb)
            x = backbone2(torch.cat((x, context[idx]), dim=1), t, c_emb)
            x = attn(x)
            outs.append(self.zero_convs[idx+1](x))
            x = downsample(x)

        x = self.mid_block1(x, t, c_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c_emb)
        outs.append(self.zero_convs[-1](x))
        return outs

class Unet(nn.Module):
    def __init__(
        self,
        dim=64,
        context_dim_factor=1,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        backbone="resnet",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.context_dim_factor = context_dim_factor

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            class_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
            )
            self.class_mlp = ConditionalEmbedding(num_labels=4,dim=dim)
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out, time_dim)),
                        get_backbone(backbone, (dim_out + int(dim_out * self.context_dim_factor), dim_out, time_dim)),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),#ATT(norm(x))+x
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = get_backbone(backbone, (mid_dim, mid_dim, time_dim))
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = get_backbone(backbone, (mid_dim, mid_dim, time_dim))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_out * 2, dim_in, time_dim)),
                        get_backbone(backbone, (dim_in, dim_in, time_dim)),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = channels
        self.final_conv = nn.Sequential(Block(dim, dim), nn.Conv2d(dim, out_dim, 1))

    def encode(self, x, t, c_emb, context):
        h = []
        for idx, (backbone, backbone2, attn, downsample) in enumerate(self.downs):
            x = backbone(x, t, c_emb)
            x = backbone2(torch.cat((x, context[idx]), dim=1), t, c_emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, c_emb)
        return x, h

    def decode(self, x, h, c_emb,  t):
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c_emb)

        for backbone, backbone2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = backbone(x, t, c_emb)
            x = backbone2(x, t, c_emb)
            x = attn(x)
            x = upsample(x)
        return self.final_conv(x)

    def forward(self, x, time=None, c_emb=None, context=None):
        t = self.time_mlp(time).reshape(-1,64) #b,64
        c_emb = self.class_mlp(c_emb)
        x, h = self.encode(x, t, c_emb, context)
        return self.decode(x, h, c_emb, t)

class ControlledUnet(Unet):
    def forward(self, x, time=None, c_emb=None, context=None, control=None):
        factor = 1
        with torch.no_grad():
            t = self.time_mlp(time)
            c_emb = self.class_mlp(c_emb)
            x, h = self.encode(x, t, c_emb, context)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t, c_emb)


        if control is not None:
            x += control.pop()*factor
        for backbone, backbone2, attn, upsample in self.ups:
            if control is None:
                x = torch.cat((x, h.pop()), dim=1)
            else: x = torch.cat((x, h.pop()+control.pop()*factor), dim=1)
            x = backbone(x, t, c_emb)
            x = backbone2(x, t, c_emb)
            x = attn(x)
            x = upsample(x)
        return self.final_conv(x)


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.exp().clamp(math.sqrt(2), 20 * math.sqrt(2))


class CondNet(nn.Module):
    def __init__(
        self,
        dim=64,  # must be the same as main net
        dim_mults=(1,2,4,8),  # must be the same as main net
        channels=9,
        backbone="resnet",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.dim_mults = dim_mults
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out)),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

    def init_state(self, shape):
        for i, ml in enumerate(self.downs):
            temp_shape = list(shape)
            temp_shape[-2] //= 2 ** i
            temp_shape[-1] //= 2 ** i
            ml[1].init_hidden(temp_shape)

    def forward(self, x):
        context = []
        for i, (resnet,  downsample) in enumerate(self.downs):
            x = resnet(x)
            context.append(x)
            x = downsample(x)
        return context
    


class controlnet3D(nn.Module):
    def __init__(
        self,
        dim=32,
        context_dim_factor=1,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        T=3,
        with_time_emb=True,
        backbone="resnet3d",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.context_dim_factor = context_dim_factor

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            class_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
            )
            self.class_mlp = ConditionalEmbedding(num_labels=4,dim=dim)
        else:
            time_dim = None
            self.time_mlp = None

        self.zero_convs = nn.ModuleList([self.make_zero_conv(self.channels)])
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out, time_dim)),
                        get_backbone(backbone, (dim_out + int(dim_out * self.context_dim_factor), dim_out, time_dim)),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),#ATT(norm(x))+x
                        Downsample3D(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            self.zero_convs.append(self.make_zero_conv(dim_out))

        mid_dim = dims[-1]
        self.mid_block1 = get_backbone(backbone, (mid_dim, mid_dim, time_dim))
        self.mid_attn = Residual(PreNorm(mid_dim*3, LinearAttention(mid_dim*3)))
        self.mid_block2 = get_backbone(backbone, (mid_dim, mid_dim, time_dim))
        self.zero_convs.append(self.make_zero_conv(mid_dim))

    def make_zero_conv(self, channels):
        return zero_module(nn.Conv3d(channels, channels, 1, padding=0))
    def init_state():
        pass

    def forward(self, x_t, control_c, time=None, c_emb=None, context=None):
        t = self.time_mlp(time)
        c_emb = self.class_mlp(c_emb)
        x =x_t + self.zero_convs[0](control_c)
        outs = []
        for idx, (backbone, backbone2, attn, downsample) in enumerate(self.downs):
            x = backbone(x, t, c_emb)
            x = backbone2(torch.cat((x, context[idx]), dim=1), t, c_emb)
            b,c,t1,h1,w1 = x.shape
            x=x.reshape(b,c*t1,h1,w1)
            x = attn(x)
            x=x.reshape(b,c,t1,h1,w1)
            outs.append(self.zero_convs[idx+1](x))
            x = downsample(x)

        x = self.mid_block1(x, t, c_emb)
        b,c,t1,h1,w1 = x.shape
        x=x.reshape(b,c*t1,h1,w1)
        x = self.mid_attn(x)
        x=x.reshape(b,c,t1,h1,w1)
        x = self.mid_block2(x, t, c_emb)
        outs.append(self.zero_convs[-1](x))
        return outs
    

class Unet3D(nn.Module):
    def __init__(
        self,
        dim=64,
        context_dim_factor=1,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        backbone="resnet3d",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.context_dim_factor = context_dim_factor

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            class_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
            )
            self.class_mlp = ConditionalEmbedding(num_labels=4,dim=dim)
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out, time_dim)),
                        get_backbone(backbone, (dim_out + int(dim_out * self.context_dim_factor), dim_out, time_dim)),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),#ATT(norm(x))+x
                        Downsample3D(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = get_backbone(backbone, (mid_dim, mid_dim, time_dim))
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = get_backbone(backbone, (mid_dim, mid_dim, time_dim))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_out * 2, dim_in, time_dim)),
                        get_backbone(backbone, (dim_in, dim_in, time_dim)),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample3D(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = channels
        self.final_conv = nn.Sequential(Block3D(dim, dim), nn.Conv3d(dim, out_dim, 1))

    def encode(self, x, t, c_emb, context):
        h = []
        for idx, (backbone, backbone2, attn, downsample) in enumerate(self.downs):
            x = backbone(x, t, c_emb)
            x = backbone2(torch.cat((x, context[idx]), dim=1), t, c_emb)
            b,c,t1,h1,w1 = x.shape
            x=x.reshape(b,c*t1,h1,w1)
            x = attn(x)
            x=x.reshape(b,c,t1,h1,w1)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, c_emb)
        return x, h

    def decode(self, x, h, c_emb,  t):
        b,c,t1,h1,w1 = x.shape
        x=x.reshape(b,c*t1,h1,w1)
        x = self.mid_attn(x)
        x=x.reshape(b,c,t1,h1,w1)
        x = self.mid_block2(x, t, c_emb)

        for backbone, backbone2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = backbone(x, t, c_emb)
            x = backbone2(x, t, c_emb)
            b,c,t1,h1,w1 = x.shape
            x=x.reshape(b,c*t1,h1,w1)
            x = attn(x)
            x=x.reshape(b,c,t1,h1,w1)
            x = upsample(x)
        return self.final_conv(x)

    def forward(self, x, time=None, c_emb=None, context=None):
        t = self.time_mlp(time).reshape(-1,64) #b,64
        c_emb = self.class_mlp(c_emb)
        x, h = self.encode(x, t, c_emb, context)
        return self.decode(x, h, c_emb, t)

class CondNet3D(nn.Module):
    def __init__(
        self,
        dim=64,  # must be the same as main net
        dim_mults=(1,2,4,8),  # must be the same as main net
        channels=9,
        backbone="resnet3d",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.dim_mults = dim_mults
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out)),
                        #ConvGRUCell(dim_out, dim_out, 3, n_layer=1),
                        Downsample3D(dim_out) if not is_last else nn.Identity(),#/2
                    ]
                )
            )

    def init_state(self, shape):
        for i, ml in enumerate(self.downs):
            temp_shape = list(shape)
            temp_shape[-2] //= 2 ** i
            temp_shape[-1] //= 2 ** i
            ml[1].init_hidden(temp_shape)

    def forward(self, x):
        context = []
        for i, (resnet,  downsample) in enumerate(self.downs):
            x = resnet(x)
            #x = conv(x)
            context.append(x)
            x = downsample(x)
        return context

class ControlledUnet3D(Unet3D):
    def forward(self, x, time=None, c_emb=None, context=None, control=None):
        factor = 1
        with torch.no_grad():
            t = self.time_mlp(time)
            c_emb = self.class_mlp(c_emb)
            x, h = self.encode(x, t, c_emb, context)
            b,c,t1,h1,w1 = x.shape
            x=x.reshape(b,c*t1,h1,w1)
            x = self.mid_attn(x)
            x=x.reshape(b,c,t1,h1,w1)
            x = self.mid_block2(x, t, c_emb)
            x = self.mid_block2(x, t, c_emb)

        if control is not None:
            x += control.pop()*factor
        for backbone, backbone2, attn, upsample in self.ups:
            if control is None:
                x = torch.cat((x, h.pop()), dim=1)
            else: x = torch.cat((x, h.pop()+control.pop()*factor), dim=1)
            x = backbone(x, t, c_emb)
            x = backbone2(x, t, c_emb)
            b,c,t1,h1,w1 = x.shape
            x=x.reshape(b,c*t1,h1,w1)
            x = attn(x)
            x=x.reshape(b,c,t1,h1,w1)
            x = upsample(x)
        return self.final_conv(x)