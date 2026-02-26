import os
import torch
import numpy as np
from torch.utils.data import DataLoader,Subset
from torch import nn
import torch.optim as optim
from radar import radarDataset
import config as cfg
from model import Ms_crn as Predictor
from draw_radar import draw,draw_all
from metrics import SAVE_MATRIC_pro
epochs = cfg.max_epoch
def test(cur_epoch):

    testData = radarDataset(root = cfg.test_data_root)
    testLoader = DataLoader(testData,batch_size=cfg.batch_size,shuffle=False)

    #gpu or cpu
    device = cfg.device
    print(device)
    
    pred_model = Predictor(size=32).to(device)

    total = sum(p.numel() for p in pred_model.parameters())
    print(total)


    l1_loss, l2_loss = nn.L1Loss().to(device), nn.MSELoss().to(device)

    epoch = cur_epoch
    model_file = ''
    print(os.path.join(cfg.ckpt_dir,model_file))
    model_info = torch.load(os.path.join(cfg.ckpt_dir,model_file),map_location='cpu')
    pred_model.load_state_dict(model_info['state_dict'])




    short_len  = cfg.in_len
    out_len = cfg.out_len

    metric = SAVE_MATRIC_pro()
    with torch.no_grad():
        pred_model.eval()
        for i,(outputs,inputs) in enumerate(testLoader):
            short_start, short_end = 0, short_len
            out_gt_start, out_gt_end = short_end, short_end+out_len
            # obtain input data and output gt
            train_data = torch.cat([inputs,outputs],dim=1).to(device)#B,T,C,H,W
            short_data = train_data[:, short_start:short_end, :, :, :]
            out_gt = train_data[:, out_gt_start:out_gt_end, :, :, :]
            out_pred,loss = pred_model(short_data,out_gt)
            loss_p2 = l1_loss(out_pred, out_gt) + l2_loss(out_pred, out_gt)
            #l = lossfunction(out_pred, out_gt)

            metric.iter(out_pred.detach().cpu().numpy().astype(np.float32)*80,out_gt.detach().cpu().numpy().astype(np.float32)*80)
       

            out_pred = out_pred.permute(1,0,2,3,4).to(cfg.device)
            out_gt = out_gt.permute(1,0,2,3,4).to(cfg.device)
                        
           

            if i%50 == 0:
                 print(str(epoch)+":test"+str(i)+"/"+str(len(testLoader)),loss_p2.item())
        
        metric.save(epoch = cur_epoch,mod=cfg.mod)

            


if __name__ == "__main__":
    test(cur_epoch=5)

