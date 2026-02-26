#@wlt

import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

import numpy as np
import pickle
from collections import defaultdict
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)




def ssim(gt, pred):

    pred = pred.reshape(-1,1,pred.shape[-2],pred.shape[-1])
    gt = gt.reshape(-1,1,gt.shape[-2],gt.shape[-1])
    cnt = 0
    ssim = 0
    for j in range(gt.shape[0]):
        gt_i = gt[j,0, :, :]
        pred_i = pred[j, 0, :, :]
        ssim += structural_similarity(gt_i, pred_i, data_range=80.0)
        cnt+=1
    return ssim/cnt


def MAE(pred, true):#CRPS
    return np.mean(np.abs(pred-true),axis=(0,1)).mean()

def MSE(pred, true):
    return np.mean((pred-true)**2,axis=(0,1)).mean()

def wMSE(pred, true):
    
    THRESHOLDS = np.array([10, 20, 30, 35, 40, 50])      
    WEIGHTS = (1, 1, 5, 10, 10, 30, 32)
    weights = numpy.ones_like(input) * WEIGHTS[0]
    for i, threshold in enumerate(THRESHOLDS):
        weights = weights + (WEIGHTS[i + 1] - WEIGHTS[i]) * (true >= threshold)
    
    return np.mean(weights*(pred-true)**2,axis=(0,1)).mean()

def PSNR(pred, true):
    mse = np.mean((np.uint8(pred)-np.uint8(true))**2)
    return 20 * np.log10(80) - 10 * np.log10(mse)

class LPIPs():
    def __init__(self):
        import lpips
        self.loss_fn_alex = lpips.LPIPS(net='alex')  
    def get(self,pred,true):
        pred = torch.from_numpy(pred.reshape(-1,1,pred.shape[-2],pred.shape[-1]))
        true = torch.from_numpy(true.reshape(-1,1,true.shape[-2],true.shape[-1]))
        return self.loss_fn_alex(pred,true).mean().item()


class CAL_CSI():
    def __init__(self) -> None:
        #init
        self.TP = [0,0,0,0,0,0]
        self.FP = [0,0,0,0,0,0]
        self.FN = [0,0,0,0,0,0]
        self.TN = [0,0,0,0,0,0]

    def iter(self, pred, true):
        for i,threshold in enumerate([10, 20, 30, 35, 40, 50]):
            pred_labels = np.ones_like(pred)
            pred_labels[pred < threshold] = 0

            true_labels = np.ones_like(true)
            true_labels[true < threshold] = 0
            TP = float(np.sum(true_labels[pred_labels == true_labels]))
            FP = float(np.sum(pred_labels == 1) - TP)
            FN = float(np.sum(true_labels == 1) - TP)
            TN = float(np.sum(true_labels[pred_labels == true_labels] == 0))


            self.TP[i] += TP
            self.FP[i] += FP
            self.FN[i] += FN
            self.TN[i] += TN
            # compute CSI, POD, FAR

        #print(self.TP[3],self.FP[3],self.FN[3],self.TN[3])
        return True
    
    def csi(self):
        ret = []
        for i,threshold in enumerate([10, 20, 30, 35, 40, 50]):
            if self.TP[i] + self.FP[i] + self.FN[i] == 0:
                # print('There is no CI')
                CSI = 0
            else:
                CSI = self.TP[i] / (self.TP[i] + self.FP[i] + self.FN[i])
            ret.append(round(CSI, 4))
        return ret

    def get_csi(self, pred, true):
        ret = []
        for i,threshold in enumerate([10, 20, 30, 35, 40, 50]):
            pred_labels = np.ones_like(pred)
            pred_labels[pred < threshold] = 0

            true_labels = np.ones_like(true)
            true_labels[true < threshold] = 0
            TP = float(np.sum(true_labels[pred_labels == true_labels]))
            FP = float(np.sum(pred_labels == 1) - TP)
            FN = float(np.sum(true_labels == 1) - TP)
            TN = float(np.sum(true_labels[pred_labels == true_labels] == 0))


            self.TP[i] += TP
            self.FP[i] += FP
            self.FN[i] += FN
            self.TN[i] += TN
            # compute CSI, POD, FAR
            if self.TP[i] + self.FP[i] + self.FN[i] == 0:
                print('There is no CI')
                CSI = 0
            else:
                CSI = self.TP[i] / (self.TP[i] + self.FP[i] + self.FN[i])
            ret.append(round(CSI, 4))
        return ret

class CAL_HSS():
    def __init__(self) -> None:
        #init
        self.TP = [0,0,0,0,0,0]
        self.FP = [0,0,0,0,0,0]
        self.FN = [0,0,0,0,0,0]
        self.TN = [0,0,0,0,0,0]

    def iter(self, pred, true):
        for i,threshold in enumerate([10, 20, 30, 35, 40, 50]):
            pred_labels = np.ones_like(pred)
            pred_labels[pred < threshold] = 0

            true_labels = np.ones_like(true)
            true_labels[true < threshold] = 0
            TP = float(np.sum(true_labels[pred_labels == true_labels]))
            FP = float(np.sum(pred_labels == 1) - TP)
            FN = float(np.sum(true_labels == 1) - TP)
            TN = float(np.sum(true_labels[pred_labels == true_labels] == 0))


            self.TP[i] += TP
            self.FP[i] += FP
            self.FN[i] += FN
            self.TN[i] += TN
            # compute CSI, POD, FAR

        #print(self.TP[3],self.FP[3],self.FN[3],self.TN[3])
        return True
    
    def hss(self):
        ret = []
        for i,threshold in enumerate([10, 20, 30, 35, 40, 50]):
            if (self.TP[i]+self.FN[i])*(self.FP[i]+self.TN[i])+(self.TP[i]+self.FP[i])*(self.FN[i]+self.TN[i]) == 0:
                # print('There is no CI')
                HSS = 0
            else:
                HSS = 2*(self.TP[i]*self.TN[i]-self.FP[i]*self.FN[i])/((self.TP[i]+self.FN[i])*(self.FP[i]+self.TN[i])+(self.TP[i]+self.FP[i])*(self.FN[i]+self.TN[i]))
            ret.append(round(HSS, 4))
        return ret

    def get_hss(self, pred, true):
        ret = []
        for i,threshold in enumerate([10, 20, 30, 35, 40, 50]):
            pred_labels = np.ones_like(pred)
            pred_labels[pred < threshold] = 0

            true_labels = np.ones_like(true)
            true_labels[true < threshold] = 0
            TP = float(np.sum(true_labels[pred_labels == true_labels]))
            FP = float(np.sum(pred_labels == 1) - TP)
            FN = float(np.sum(true_labels == 1) - TP)
            TN = float(np.sum(true_labels[pred_labels == true_labels] == 0))


            self.TP[i] += TP
            self.FP[i] += FP
            self.FN[i] += FN
            self.TN[i] += TN
            # compute CSI, POD, FAR
            if (self.TP[i]+self.FN[i])*(self.FP[i]+self.TN[i])+(self.TP[i]+self.FP[i])*(self.FN[i]+self.TN[i]) == 0:
                # print('There is no CI')
                HSS = 0
            else:
                HSS = 2*(self.TP[i]*self.TN[i]-self.FP[i]*self.FN[i])/((self.TP[i]+self.FN[i])*(self.FP[i]+self.TN[i])+(self.TP[i]+self.FP[i])*(self.FN[i]+self.TN[i]))
            ret.append(round(HSS, 4))
        return ret
    

class CAL_FAR():
    def __init__(self) -> None:
        #init
        self.TP = [0,0,0,0,0,0]
        self.FP = [0,0,0,0,0,0]
        self.FN = [0,0,0,0,0,0]
        self.TN = [0,0,0,0,0,0]

    def iter(self, pred, true):
        for i,threshold in enumerate([10, 20, 30, 35, 40, 50]):
            pred_labels = np.ones_like(pred)
            pred_labels[pred < threshold] = 0

            true_labels = np.ones_like(true)
            true_labels[true < threshold] = 0
            TP = float(np.sum(true_labels[pred_labels == true_labels]))
            FP = float(np.sum(pred_labels == 1) - TP)
            FN = float(np.sum(true_labels == 1) - TP)
            TN = float(np.sum(true_labels[pred_labels == true_labels] == 0))


            self.TP[i] += TP
            self.FP[i] += FP
            self.FN[i] += FN
            self.TN[i] += TN
            # compute CSI, POD, FAR

        #print(self.TP[3],self.FP[3],self.FN[3],self.TN[3])
        return True
    
    def far(self):
        ret = []
        for i,threshold in enumerate([10, 20, 30, 35, 40, 50]):
            if self.FP[i] + self.TP[i]== 0:
                # print('There is no CI')
                FAR = 0
            else:
                FAR = self.FP[i] / (self.TP[i] + self.FP[i])
            ret.append(round(FAR, 4))
        return ret

    def get_far(self, pred, true):
        ret = []
        for i,threshold in enumerate([10, 20, 30, 35, 40, 50]):
            pred_labels = np.ones_like(pred)
            pred_labels[pred < threshold] = 0

            true_labels = np.ones_like(true)
            true_labels[true < threshold] = 0
            TP = float(np.sum(true_labels[pred_labels == true_labels]))
            FP = float(np.sum(pred_labels == 1) - TP)
            FN = float(np.sum(true_labels == 1) - TP)
            TN = float(np.sum(true_labels[pred_labels == true_labels] == 0))


            self.TP[i] += TP
            self.FP[i] += FP
            self.FN[i] += FN
            self.TN[i] += TN
            # compute CSI, POD, FAR
            if self.FP[i] + self.TP[i]== 0:
                # print('There is no CI')
                FAR = 0
            else:
                FAR = self.FP[i] / (self.TP[i] + self.FP[i])
            ret.append(round(FAR, 4))
        return ret

class CAL_FBIAS():  
    def __init__(self) -> None:

        self.TP = [0,0,0,0,0,0]
        self.FP = [0,0,0,0,0,0]
        self.FN = [0,0,0,0,0,0]
        self.TN = [0,0,0,0,0,0]

    def iter(self, pred, true):  
        for i, threshold in enumerate([10, 20, 30, 35, 40, 50]):
            pred_labels = np.where(pred >= threshold, 1, 0)
            true_labels = np.where(true >= threshold, 1, 0)

            TP = np.sum((pred_labels == 1) & (true_labels == 1))
            FP = np.sum((pred_labels == 1) & (true_labels == 0))
            FN = np.sum((pred_labels == 0) & (true_labels == 1))
            TN = np.sum((pred_labels == 0) & (true_labels == 0))

            self.TP[i] += TP
            self.FP[i] += FP
            self.FN[i] += FN
            self.TN[i] += TN

        return True

    def fbias(self):  
        ret = []
        for i, threshold in enumerate([10, 20, 30, 35, 40, 50]):
            H = self.TP[i]
            F = self.FP[i]
            M = self.FN[i]
            if (H + M) == 0:
                bias = np.nan  
            else:
                bias = (H + F) / (H + M)
            ret.append(round(bias, 4) if not np.isnan(bias) else None)
        return ret

    def get_fbias(self, pred, true):  # 单次计算（非累积）
        ret = []
        for i, threshold in enumerate([10, 20, 30, 35, 40, 50]):
            pred_labels = np.where(pred >= threshold, 1, 0)
            true_labels = np.where(true >= threshold, 1, 0)

            H = np.sum((pred_labels == 1) & (true_labels == 1))
            F = np.sum((pred_labels == 1) & (true_labels == 0))
            M = np.sum((pred_labels == 0) & (true_labels == 1))

            if (H + M) == 0:
                bias = np.nan
            else:
                bias = (H + F) / (H + M)
            ret.append(round(bias, 4) if not np.isnan(bias) else None)
        return ret


class SAVE_MATRIC_pro():
    def __init__(self,work_path=None,out_seq_len = 12) -> None:
        self.work_path = work_path
        self.out_seq_len = out_seq_len
        self.ps = 4
        self.pool = nn.MaxPool2d(kernel_size=self.ps,stride=self.ps)
        self.index = []
        self.cal_csi = []
        self.cal_csi_pool = []
        self.cal_hss = []
        self.cal_far = []
        self.cal_fbias = []
        for i in range(out_seq_len):
            self.index.append(i)
            self.cal_csi.append(CAL_CSI())
            self.cal_csi_pool.append(CAL_CSI())
            self.cal_hss.append(CAL_HSS())
            self.cal_far.append(CAL_FAR())
            self.cal_fbias.append(CAL_FBIAS())
        self.cnt = 0
        #以下按样本 T 平均
        self.mMAE = 0
        self.mMSE = 0
        self.wMSE = 0
        self.mPSNR = 0
        self.mSSIM = 0
        self.mLPIPS = 0
        self.lpipsfunc = LPIPs()
        
    def iter(self,pred,true):
        
        #mae
        self.cnt += 1
        self.mMAE +=MAE(pred,true)
        self.mMSE +=MSE(pred,true)
        self.wMSE += wMSE(pred,true)
        self.mPSNR += PSNR(pred,true)
        self.mSSIM += ssim(pred,true)
        self.mLPIPS += self.lpipsfunc.get(pred,true)

        #csi
        for i,t in enumerate(self.index):
            self.cal_csi[i].iter(pred[:,t,:,:,:],true[:,t,:,:,:])
            self.cal_hss[i].iter(pred[:,t,:,:,:],true[:,t,:,:,:])
            self.cal_far[i].iter(pred[:,t,:,:,:],true[:,t,:,:,:])
            self.cal_fbias[i].iter(pred[:,t,:,:,:],true[:,t,:,:,:])
        #pool csi
        if self.ps!=1:
            B,T,C,H,W = pred.shape
            pred = np.array(self.pool(torch.from_numpy(pred.reshape(-1,H,W))).reshape(B,T,C,H//self.ps,W//self.ps))
            true = np.array(self.pool(torch.from_numpy(true.reshape(-1,H,W))).reshape(B,T,C,H//self.ps,W//self.ps))
        for i,t in enumerate(self.index):
            self.cal_csi_pool[i].iter(pred[:,t,:,:,:],true[:,t,:,:,:])

        



    def save(self,epoch=1,mod=''):
        #save csi hss
        index = ['10dbz','20dbz','30dbz','35dbz','40dbz','50dbz']
        with open(self.work_path+"/metric.txt", 'a') as f:
            f.write("epoch:{}-{}---------------------------------\n".format(epoch,mod))
            self.csi = []
            self.hss = []
            self.far = []
            self.fbias = []
        #save csi
            for t in range(self.out_seq_len):
                self.csi.append(self.cal_csi[t].csi())
                self.hss.append(self.cal_hss[t].hss())
                self.far.append(self.cal_far[t].far())
                self.fbias.append(self.cal_fbias[t].fbias())
            sum = 0
            for i,throhold in enumerate(index):
                f.write("{}:".format(throhold))
                temp = 0
                for t in range(self.out_seq_len):
                    f.write("{},".format(str(self.csi[t][i])))
                    temp += self.csi[t][i]
                sum+=temp/self.out_seq_len if (throhold in ['10dbz','20dbz','30dbz','35dbz','40dbz']) else 0#
                f.write("M_{}\n".format(str(round(temp/self.out_seq_len,4))))
            f.write("csi_mean_{}\n".format(str(round(sum/5.0,4))))
        #save csi_pool
            self.csi = []
            for t in range(self.out_seq_len):
                self.csi.append(self.cal_csi_pool[t].csi())
            sum = 0
            for i,throhold in enumerate(index):
                f.write("{}:".format(throhold))
                temp = 0
                for t in range(self.out_seq_len):
                    f.write("{},".format(str(self.csi[t][i])))
                    temp += self.csi[t][i]
                sum+=temp/self.out_seq_len if (throhold in ['10dbz','20dbz','30dbz','35dbz','40dbz']) else 0#
                f.write("M_{}".format(str(round(temp/self.out_seq_len,4))))
                f.write("\n")
            f.write("csi_pool_mean_{}\n".format(str(round(sum/5.0,4))))
        #save hss
            sum = 0
            for i,throhold in enumerate(index):
                f.write("{}:".format(throhold))
                temp = 0
                for t in range(self.out_seq_len):
                    f.write("{},".format(str(self.hss[t][i])))
                    temp += self.hss[t][i]
                sum+=temp/self.out_seq_len if (throhold in ['10dbz','20dbz','30dbz','35dbz','40dbz']) else 0#
                f.write("M_{}\n".format(str(round(temp/self.out_seq_len,4))))
            f.write("hss_mean_{}\n".format(str(round(sum/5.0,4))))
        #save hss
            sum = 0
            for i,throhold in enumerate(index):
                f.write("{}:".format(throhold))
                temp = 0
                for t in range(self.out_seq_len):
                    f.write("{},".format(str(self.far[t][i])))
                    temp += self.far[t][i]
                sum+=temp/self.out_seq_len if (throhold in ['10dbz','20dbz','30dbz','35dbz','40dbz']) else 0#
                f.write("M_{}\n".format(str(round(temp/self.out_seq_len,4))))
            f.write("far_mean_{}\n".format(str(round(sum/5.0,4))))
         #save fbias
            sum = 0
            for i,throhold in enumerate(index):
                f.write("{}:".format(throhold))
                temp = 0
                for t in range(self.out_seq_len):
                    f.write("{},".format(str(self.fbias[t][i])))
                    temp += self.fbias[t][i]
                sum+=temp/self.out_seq_len if (throhold in ['10dbz','20dbz','30dbz','35dbz','40dbz']) else 0#
                f.write("M_{}\n".format(str(round(temp/self.out_seq_len,4))))
            f.write("fbias_mean_{}\n".format(str(round(sum/5.0,4))))
        #save mae
            f.write("mae(crps):{}\n".format(self.mMAE/self.cnt))
            f.write("mse:{}\n".format(self.mMSE/self.cnt))
            f.write("wmse:{}\n".format(self.wMSE/self.cnt))
        #save psnr
            f.write("PSNR:{}\n".format(self.mPSNR/self.cnt))
        #save ssim
            f.write("SSIM:{}\n".format(self.mSSIM/self.cnt))
        #save lpips
            f.write("lpips:{}\n".format(self.mLPIPS/self.cnt))



import numpy as np
from scipy import stats
from collections import defaultdict

class CSISigMetric2():
    def __init__(self, work_path = None, thresholds=[10,20,30,35,40]) -> None:
        if work_path == None:
            print("workpath none")
            return
        self.work_path = work_path
        self.thresholds = thresholds
        self.thresholds_with_mean = thresholds + ["mean"]
        self.records = {th: [] for th in self.thresholds_with_mean}  # {th: [sample1, sample2,...]}

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.records_pool = {th: [] for th in self.thresholds_with_mean} 

    def update(self, pred, gt):
        """
        pred, gt: numpy array, shape = (B, H, W)
        """
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        B = pred.shape[0]

        for i in range(B):
            sample_csi = []
            for th in self.thresholds:
                p_bin = pred[i] >= th
                g_bin = gt[i] >= th
                TP = np.logical_and(p_bin, g_bin).sum()
                FP = np.logical_and(p_bin, ~g_bin).sum()
                FN = np.logical_and(~p_bin, g_bin).sum()
                denom = TP + FP + FN
                csi = TP/denom if denom>0 else np.nan
                self.records[th].append(csi)
                sample_csi.append(csi)
            # 平均阈值
            mean_csi = np.nanmean(sample_csi)
            self.records["mean"].append(mean_csi)
        
        B,T,C,H,W = pred.shape
        self.ps = 2
        pred = np.array(self.pool(torch.from_numpy(pred.reshape(-1,H,W))).reshape(B,T,C,H//self.ps,W//self.ps))
        gt = np.array(self.pool(torch.from_numpy(gt.reshape(-1,H,W))).reshape(B,T,C,H//self.ps,W//self.ps))
        for i in range(B):
            sample_csi_pool = []
            for th in self.thresholds:
                p_bin = pred[i] >= th
                g_bin = gt[i] >= th
                TP = np.logical_and(p_bin, g_bin).sum()
                FP = np.logical_and(p_bin, ~g_bin).sum()
                FN = np.logical_and(~p_bin, g_bin).sum()
                denom = TP + FP + FN
                csi = TP/denom if denom>0 else np.nan
                self.records_pool[th].append(csi)
                sample_csi_pool.append(csi)
            mean_csi = np.nanmean(sample_csi_pool)
            self.records_pool["mean"].append(mean_csi)

    def reset(self):
        self.records = {th: [] for th in self.thresholds_with_mean}
        self.records_pool = {th: [] for th in self.thresholds_with_mean}



    def save(self, filepath = None):
        if filepath == None:
            filepath = self.work_path
        with open(filepath+"/CSI_records.pkl", "wb") as f:
            pickle.dump(self.records, f)
        with open(filepath+"/CSI_pool_records.pkl", "wb") as f:
            pickle.dump(self.records_pool, f)

    def load(self, filepath =None):
        if filepath == None:
            filepath = self.work_path
        with open(filepath+"/CSI_records.pkl", "rb") as f:
            self.records = pickle.load(f)
        with open(filepath+"/CSI_pool_records.pkl", "rb") as f:
            self.records_pool = pickle.load(f)

    def bootstrap_ci(self, data, n_bootstrap=1000, ci=95):
        data = np.array(data)
        data = data[~np.isnan(data)]
        n = len(data)
        if n == 0:
            return np.nan, np.nan, np.nan
        boot_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            boot_means.append(np.mean(sample))
        low = np.percentile(boot_means, (100 - ci) / 2)
        high = np.percentile(boot_means, 100 - (100 - ci) / 2)
        return np.mean(data), np.std(data, ddof=1), (low, high), n


    def compute_statistics(self, if_bs = False):

        summary = {}
        for th, values in self.records.items():
            if if_bs:
                mean_, std_, ci95, n = self.bootstrap_ci(values, n_bootstrap=2000, ci=95)
                summary[th] = {
                    "mean": mean_,
                    "std": std_,
                    "ci95": ci95,
                    "n": n
                }
            else:
                arr = np.array(values)
                arr = arr[~np.isnan(arr)]
                n = len(arr)
                if n == 0:
                    continue
                mean_ = np.mean(arr)
                std_ = np.std(arr, ddof=1)
                # 95% CI
                ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean_, scale=std_/np.sqrt(n))
                summary[th] = {
                    "mean": mean_,
                    "std": std_,
                    "ci95": (ci_low, ci_high),
                    "n": n
                }
        return summary

    
    def compute_statistics_pool(self, if_bs = False):
        summary = {}
        for th, values in self.records_pool.items():
            if if_bs:
                mean_, std_, ci95, n = self.bootstrap_ci(values, n_bootstrap=2000, ci=95)
                summary[th] = {
                    "mean": mean_,
                    "std": std_,
                    "ci95": ci95,
                    "n": n
                }
            else:
                arr = np.array(values)
                arr = arr[~np.isnan(arr)]
                n = len(arr)
                if n == 0:
                    continue
                mean_ = np.mean(arr)
                std_ = np.std(arr, ddof=1)
                # 95% CI
                ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean_, scale=std_/np.sqrt(n))
                summary[th] = {
                    "mean": mean_,
                    "std": std_,
                    "ci95": (ci_low, ci_high),
                    "n": n
                }
        return summary
    

    @staticmethod
    def pairwise_significance(all_records, method_names=None, use_wilcoxon=False, alpha=0.05):
            if method_names is None:
                method_names = list(all_records.keys())
            thresholds = all_records[method_names[0]].thresholds_with_mean
            results = defaultdict(dict)

            for th in thresholds:
                for m1, m2 in combinations(method_names, 2):
                    a = np.array(all_records[m1].records[th])
                    b = np.array(all_records[m2].records[th])
                    mask = ~np.isnan(a) & ~np.isnan(b)
                    if use_wilcoxon:
                        if np.sum(mask) == 0:
                            stat, p = np.nan, np.nan
                        else:
                            stat, p = stats.wilcoxon(a[mask], b[mask])
                    else:
                        if np.sum(mask) == 0:
                            stat, p = np.nan, np.nan
                        else:
                            stat, p = stats.ttest_rel(a[mask], b[mask])
                    sig = "*" if p is not None and p < alpha else ""
                    results[(m1, m2)][th] = {"stat": stat, "p": p, "sig": sig}
            return results
    
    @staticmethod
    def pairwise_significance_pool(all_records, method_names=None, use_wilcoxon=False, alpha=0.05):
            if method_names is None:
                method_names = list(all_records.keys())
            thresholds = all_records[method_names[0]].thresholds_with_mean
            results = defaultdict(dict)

            for th in thresholds:
                for m1, m2 in combinations(method_names, 2):
                    a = np.array(all_records[m1].records_pool[th])
                    b = np.array(all_records[m2].records_pool[th])
                    mask = ~np.isnan(a) & ~np.isnan(b)
                    if use_wilcoxon:

                        if np.sum(mask) == 0:
                            stat, p = np.nan, np.nan
                        else:
                            stat, p = stats.wilcoxon(a[mask], b[mask])
                    else:
                        if np.sum(mask) == 0:
                            stat, p = np.nan, np.nan
                        else:
                            stat, p = stats.ttest_rel(a[mask], b[mask])
                    sig = "*" if p is not None and p < alpha else ""
                    results[(m1, m2)][th] = {"stat": stat, "p": p, "sig": sig}
            return results

    def get_sample_csi(self, sample_idx):
        sample_dict = {}
        for th, arr in self.records.items():
            if sample_idx < len(arr):
                sample_dict[th] = arr[sample_idx]
            else:
                sample_dict[th] = np.nan  
        return sample_dict
    
    def get_sample_csi_pool(self, sample_idx):
        sample_dict = {}
        for th, arr in self.records_pool.items():
            if sample_idx < len(arr):
                sample_dict[th] = arr[sample_idx]
            else:
                sample_dict[th] = np.nan  
        return sample_dict
