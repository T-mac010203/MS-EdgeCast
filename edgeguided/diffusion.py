
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from .model import Unet,CondNet,ControlledUnet,controlnet,Unet3D,CondNet3D,ControlledUnet3D,controlnet3D
import copy
# from draw_radar import draw





def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.from_numpy(np.clip(betas, a_min=0, a_max=0.999))


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class sobel_edge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(1,1,1,1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(1,1,1,1)

        self.weight_x = torch.nn.Parameter(sobel_x, requires_grad=False)
        self.weight_y = torch.nn.Parameter(sobel_y, requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    def forward(self,x):
        edge_x = F.conv2d(x, self.weight_x, self.bias, padding=1)
        edge_y = F.conv2d(x, self.weight_y, self.bias, padding=1)

        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)

        edges = (edges - edges.min()) / (edges.max() - edges.min())

        return edges



class GaussianDiffusionTrainer(nn.Module):
    def __init__(self,  T):
        super().__init__()

        self.model = Unet()
        self.cond_model = CondNet()

        self.T = T
        self.betas = cosine_beta_schedule(T)
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))#sqrt(a)
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))#sqrt(1-a)
        
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))


    def forward(self,cond,x_0,c_emb):
        """
        Algorithm 1.
        """
        cond = self.cond_model(cond)

        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        y_0 = x_0
        y_t = (
            extract(self.sqrt_alphas_bar, t, y_0.shape) * y_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, y_0.shape) * noise)
        
        loss = F.mse_loss(self.model(y_t, t,c_emb,cond),noise)
        
        return loss
    
    @torch.no_grad()
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )
    @torch.no_grad()
    def p_mean_variance(self, x_t, t, c_emb, cond):
        # below: only log_variance is used in the KL computations
        #var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = self.posterior_var
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t, c_emb,cond)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    @torch.no_grad()
    def sample(self, cond, c_emb):
        """
        Algorithm 2.
        """
        y_t = torch.randn_like(cond[:,6:18,:,:])
        cond = self.cond_model(cond)

        for time_step in reversed(range(self.T)):
            t = y_t.new_ones([y_t.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=y_t, t=t, c_emb=c_emb, cond=cond)
            if time_step > 0:
                noise = torch.randn_like(y_t)
            else:
                noise = 0
            y_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(y_t).int().sum() == 0, "nan in tensor."

        generated_frame = y_t 
        return generated_frame
    
    
class Edge_guided_diffusion(nn.Module):
    def __init__(self, T, prepath = None,path =None):
        super().__init__()
        self.model = ControlledUnet()
        self.cond_model = CondNet()

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.cond_model.parameters():
            param.requires_grad = False

        self.control_model = controlnet()
        self.T = T
        self.betas = cosine_beta_schedule(T)
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))#sqrt(a)
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))#sqrt(1-a)
        
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        
        self.init_state(prepath = prepath,path = path)

        self.edge = sobel_edge()

    def init_state(self,prepath,path):
       
        model_info = torch.load(prepath,map_location='cpu')
        
        self.load_state_dict(model_info['state_dict'], strict=False)
        if path is None:
            param = copy.deepcopy(self.model.state_dict())
            self.control_model.load_state_dict(model_info['state_dict'], strict = False)
            print(set(self.control_model.state_dict().keys()) - set(param.keys()))

        else:
            model_info = torch.load(os.path.join("",path),map_location='cpu')
            self.control_model.load_state_dict(model_info['state_dict'])

            pretrained_keys = set(model_info['state_dict'].keys())
            model_keys = set(self.control_model.state_dict().keys())
        
           
            missing_keys = model_keys - pretrained_keys
            unexpected_keys = pretrained_keys - model_keys
            
            strict = True
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                print(f"Missing keys: {missing_keys}")
                print(f"Unexpected keys: {unexpected_keys}")
                if strict:
                    raise ValueError("Some keys are not matched")
            else:
                print("All keys are matched")
            

    def forward(self,cond,x_0,c_emb):
        
        b1,t1,h1,w1 = x_0.shape

        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        y_0 = x_0
        y_t = (
            extract(self.sqrt_alphas_bar, t, y_0.shape) * y_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, y_0.shape) * noise)
        
        context = self.cond_model(cond)
    

        edges = self.edge(cond[:,6:9,:,:].reshape(-1,1,h1,w1)).reshape(-1,t1,h1,w1)

        control_x = self.control_model(y_t,edges,t,c_emb,context)
        
        loss = F.mse_loss(self.model(y_t, t, c_emb, context, control_x),noise)
        return loss
    
    @torch.no_grad()
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )
    @torch.no_grad()
    def p_mean_variance(self, x_t, t,c_emb, cond,edges):
        # below: only log_variance is used in the KL computations
        #var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = self.posterior_var
        var = extract(var, t, x_t.shape)

        context = self.cond_model(cond)
        control_x = self.control_model(x_t,edges,t,c_emb,context)
        #control_x = None
        eps = self.model(x_t, t,c_emb, context, control_x)
        

        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    @torch.no_grad()
    def sample(self, cond, c_emb):
        """
        Algorithm 2.
        """
        y_t = torch.randn_like(cond[:,6:9,:,:])
        b1,t1,h1,w1 = y_t.shape
        edges = self.edge(cond[:,6:9,:,:].reshape(-1,1,h1,w1)).reshape(-1,t1,h1,w1)
        for time_step in reversed(range(self.T)):
            
            print(time_step)
            t = y_t.new_ones([y_t.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=y_t, t=t, c_emb=c_emb, cond=cond, edges=edges)

            if time_step > 0:
                noise = torch.randn_like(y_t)
            else:
                noise = 0
            y_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(y_t).int().sum() == 0, "nan in tensor."

        generated_frame = y_t 

        return generated_frame


class GaussianDiffusion3DTrainer(nn.Module):
    def __init__(self,  T):
        super().__init__()
        self.model = Unet3D()#.to(device)
        self.cond_model = CondNet3D()#.to(device)

        self.T = T
        self.betas = cosine_beta_schedule(T)
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))#sqrt(a)
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))#sqrt(1-a)
        
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def forward(self,cond,x_0,c_emb):
        """
        Algorithm 1.
        """
        cond = self.cond_model(cond)

        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        y_0 = x_0
        y_t = (
            extract(self.sqrt_alphas_bar, t, y_0.shape) * y_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, y_0.shape) * noise)
        
        loss = F.mse_loss(self.model(y_t, t,c_emb,cond),noise)
        
        return loss
    
    @torch.no_grad()
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )
    @torch.no_grad()
    def p_mean_variance(self, x_t, t, c_emb, cond):
        var = self.posterior_var
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t, c_emb,cond)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    @torch.no_grad()
    def sample(self, cond, c_emb):
        """
        Algorithm 2.
        """
        y_t = torch.randn_like(cond[:,:,6:18,:,:])
        cond_feat = self.cond_model(cond)

        for time_step in reversed(range(self.T)):
            #print(time_step)
            t = y_t.new_ones([y_t.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=y_t, t=t, c_emb=c_emb, cond=cond_feat.copy())
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(y_t)
            else:
                noise = 0
            y_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(y_t).int().sum() == 0, "nan in tensor."

        generated_frame = y_t 
        return generated_frame
    



class Edge_guided_diffusion3d(nn.Module):
    def __init__(self, T, prepath = None,path =None):
        super().__init__()
        self.model = ControlledUnet3D()
        self.cond_model = CondNet3D()

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.cond_model.parameters():
            param.requires_grad = False

        self.control_model = controlnet3D()
        self.T = T
        self.betas = cosine_beta_schedule(T)
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))#sqrt(a)
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))#sqrt(1-a)
        
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        
        self.init_state(prepath = prepath,path = path)

        self.edge = sobel_edge()

    def init_state(self,prepath,path):
        if prepath is not None:
            model_info = torch.load(prepath,map_location='cpu')
            self.load_state_dict(model_info['state_dict'], strict=False)

        # else:
        #     assert("")
        if path is None:
            param = copy.deepcopy(self.model.state_dict())
            self.control_model.load_state_dict(model_info['state_dict'], strict = False)
            print(set(self.control_model.state_dict().keys()) - set(param.keys()))

        else:
            model_info = torch.load(os.path.join("",path),map_location='cpu')
            self.control_model.load_state_dict(model_info['state_dict'])

            pretrained_keys = set(model_info['state_dict'].keys())
            model_keys = set(self.control_model.state_dict().keys())
        
           
            missing_keys = model_keys - pretrained_keys
            unexpected_keys = pretrained_keys - model_keys
            
            strict = True
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                print(f"Missing keys: {missing_keys}")
                print(f"Unexpected keys: {unexpected_keys}")
                if strict:
                    raise ValueError("Some keys are not matched")
            else:
                print("All keys are matched")
            

    def forward(self,cond,x_0,c_emb):
        
        b1,c1,t1,h1,w1 = x_0.shape

        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        y_0 = x_0
        y_t = (
            extract(self.sqrt_alphas_bar, t, y_0.shape) * y_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, y_0.shape) * noise)
        
        context = self.cond_model(cond)
    

        edges = self.edge(cond[:,:,6:9,:,:].reshape(-1,1,h1,w1)).reshape(-1,1,t1,h1,w1)

        control_x = self.control_model(y_t,edges,t,c_emb,context)
        
        loss = F.mse_loss(self.model(y_t, t, c_emb, context, control_x),noise)
        return loss
    
    @torch.no_grad()
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )
    @torch.no_grad()
    def p_mean_variance(self, x_t, t,c_emb, cond,edges):
        # below: only log_variance is used in the KL computations
        #var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = self.posterior_var
        var = extract(var, t, x_t.shape)

        context = self.cond_model(cond)
        control_x = self.control_model(x_t,edges,t,c_emb,context)
        #control_x = None
        eps = self.model(x_t, t,c_emb, context, control_x)
        

        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    @torch.no_grad()
    def sample(self, cond, c_emb):
        """
        Algorithm 2.
        """
        y_t = torch.randn_like(cond[:,:,6:9,:,:])
        b1,c1,t1,h1,w1 = y_t.shape
        edges = self.edge(cond[:,:,6:9,:,:].reshape(-1,1,h1,w1)).reshape(-1,1,t1,h1,w1)
        for time_step in reversed(range(self.T)):
            
            print(time_step)
            t = y_t.new_ones([y_t.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=y_t, t=t, c_emb=c_emb, cond=cond, edges=edges)

            if time_step > 0:
                noise = torch.randn_like(y_t)
            else:
                noise = 0
            y_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(y_t).int().sum() == 0, "nan in tensor."

        generated_frame = y_t 

        return generated_frame
    
