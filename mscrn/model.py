import torch
import torch.nn.functional as F
import torch.nn as nn
from .modules import ConvGRU,ResBlock,wrap,Upsample,Downsample,unfold32,AttnBlock,fold32
    
class Patch_crn(nn.Module):
    def __init__(self,size = 32,H = 1024) -> None:
        super().__init__()
        c,h,w = 1,size,size
        self.size = size
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU())#8,8,128
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(3, 3), stride=2, padding=1, output_padding=1))

        self.convgru_encode = nn.ModuleList(
            [ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0),
            ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0),
            ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0),
            ])
        self.convgru_forcast = nn.ModuleList(
            [ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0),
            ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0),
            ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0)])
        self.unfold = unfold32(size=size)
        self.fold =fold32(size=size,H=H)
    def forward(self,x,out_gt,out_len):
        _,_,_,H_,W_ = x.shape
        x = self.unfold(x)

        batch_size,input_len,C,H,W = x.shape


        x = self.spatial_encoder(x.reshape(-1,1,H,W))
        x = x.reshape(-1,input_len,x.size()[1],x.size()[2],x.size()[3])


        h,outputs_h =[], x
        for layer_i in range(3):
            h.append(0)
            outputs_h,h[layer_i]=self.convgru_encode[layer_i](outputs_h)

        outputs_h = torch.randn_like(torch.empty([batch_size, out_len,128, H//4,W//4])).to(x.device)
        for layer_i in range(3)[::-1]:
            outputs_h,_ = self.convgru_forcast[layer_i](outputs_h,h[layer_i])
        
        out_pred = self.spatial_decoder(outputs_h.reshape(-1,128,self.size//4,self.size//4)).reshape(-1,12,1,self.size,self.size)

        out_pred = self.fold(out_pred)

        return out_pred

class flow_crn(nn.Module):
    def __init__(self,H) -> None:
        super().__init__()
        h = w = H
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU())
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=(3, 3), stride=2, padding=1, output_padding=1))

        self.convgru_encode = nn.ModuleList(
            [ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0),
            ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0),
            ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0),
            ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0)])
        self.convgru_forcast = nn.ModuleList(
            [ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0),
            ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0),
            ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0),
            ConvGRU(in_channels=128,out_channels=128,kernel_size=[3,3],out_shape = [128,h//4,w//4],mode = 0)])
        self.loss = nn.MSELoss()
    def forward(self, x, out_gt, out_len):
        batch_size,input_len,C,H,W = x.shape
        x0 = x[:,-1,:,:,:]


        x = self.spatial_encoder(x.reshape(-1,1,H,W))
        x = x.reshape(-1,input_len,x.size()[1],x.size()[2],x.size()[3])

        h,outputs_h =[], x
        for layer_i in range(4):
            h.append(0)
            outputs_h,h[layer_i]=self.convgru_encode[layer_i](outputs_h)

        outputs_h = torch.randn_like(torch.empty([batch_size, out_len,128, H//4,W//4])).to(x.device)
        for layer_i in range(4)[::-1]:
            outputs_h,_ = self.convgru_forcast[layer_i](outputs_h,h[layer_i])
        
        out_pred_motion = self.spatial_decoder(outputs_h.reshape(-1,128,H//4,W//4)).reshape(-1,out_len,2,H,W)
        out_pred_motion = out_pred_motion.transpose(0,1)

        out_pred = []
        for t in range(out_len):
            if t == 0:
                out_pred.append(wrap(x0,out_pred_motion[t]))
            else: 
                out_pred.append(wrap(out_pred[t-1],out_pred_motion[t]))
        
        out_pred = torch.stack(out_pred)
        out_pred = out_pred.transpose(dim0=0, dim1=1)
        wloss = self.loss(out_pred,out_gt)

        return out_pred,wloss

class Ms_crn(nn.Module):
    def __init__(self,size,h) -> None:
        super().__init__()
        self.patch = Patch_crn(size=size,H=h)
        self.flow = flow_crn(H=h)
        self.patch_down = nn.Sequential(
        ResBlock(12,24),
        Downsample(24,48),
        ResBlock(48,48),
        Downsample(48,96))
        self.flow_down = nn.Sequential(
        ResBlock(12,24),
        Downsample(24,48),
        ResBlock(48,48),
        Downsample(48,96))
        self.cross_att = AttnBlock(96)#96,h//4,w//4
        self.fusion = nn.ModuleList([
        ResBlock(96,96),
        Upsample(96,48),
        ResBlock(48,48),
        Upsample(48,24),
        nn.Conv2d(24,12,3,1,1)
        ])
        self.loss = nn.MSELoss()
    def forward(self, x, out_gt=None):
        B,out_len,C,H,W = out_gt.shape
        evo_x, evo_loss = self.flow(x, out_gt, out_len)
        patch_x = self.patch(x, out_gt, out_len)

        evo_x = self.flow_down(evo_x.reshape(-1,out_len,H,W))
        patch_x = self.patch_down(patch_x.reshape(-1,out_len,H,W))
        x = self.cross_att(evo_x,patch_x)
    
        for block in self.fusion:
            x = block(x)

        fusion_loss = self.loss(x.reshape(B,out_len,C,H,W),out_gt)
        loss = fusion_loss+0.001*evo_loss
        return x,loss
