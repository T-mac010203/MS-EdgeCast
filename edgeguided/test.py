import datetime
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import config as cfg
from radar import radarDataset_evo
from diffusion import Edge_guided_diffusion
from metrics import CSISigMetric2,SAVE_MATRIC_pro
import datetime
epochs = cfg.max_epoch
def teat_all():

    testData = radarDataset_evo(root = cfg.test_data_root)
    testLoader = DataLoader(testData,batch_size=cfg.batch_size,shuffle=False)

    
    #gpu or cpu
    device = cfg.device
    print(device)
    diffusion = Edge_guided_diffusion(1000,prepath="",path="").to(device)

    total = sum(p.numel() for p in diffusion.parameters())
    print(total)

    diffusion.eval()

    metric = SAVE_MATRIC_pro(work_path=cfg.work_path)
    sig_metric = CSISigMetric2(work_path=cfg.work_path)
    for i,(outputs,inputs) in enumerate(testLoader):
        outputs = outputs.to(cfg.device)
        inputs = inputs.to(cfg.device)
        predictions = []
        print(datetime.datetime.now())
        
        for c_emb in range(0,4):
            inputs_p = torch.cat([inputs[:,:6,:,:],inputs[:,6+c_emb*3:6+c_emb*3+3,:,:]],dim=1).squeeze(2)
            with torch.no_grad():
                c_tensor = torch.full(
                    (inputs.shape[0],),  # B
                    fill_value=c_emb,
                    dtype=torch.long,
                    device=device
                    )
                outpred = diffusion.sample(inputs_p,c_emb=c_tensor)
            predictions.append(outpred)
            
        i+=1
        outpred = torch.cat(predictions,dim=1).unsqueeze(2)

        metric.iter(outpred.detach().cpu().numpy().astype(np.float32)*80,outputs.detach().cpu().numpy().astype(np.float32)*80)
        sig_metric.update(outpred.detach().cpu().numpy().astype(np.float32)*80,outputs.detach().cpu().numpy().astype(np.float32)*80)
        
       



        if i%5 ==0:
            current_time = datetime.datetime.now()
            print(str(current_time)+'_'+str(i), "/", len(testLoader))


    print(sig_metric.compute_statistics())
    print(sig_metric.compute_statistics_pool())



if __name__ == "__main__": 
    teat_all()
