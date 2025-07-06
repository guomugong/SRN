import numpy as np
import torch
import cv2
import torch.nn as nn
from model.unet_model import UNet
from utils.dataset import FundusSeg_Loader
from utils.eval_metrics import perform_metrics,cal_f1
import copy
import sys 
from sklearn.metrics import roc_auc_score
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

source_domain=sys.argv[1]
target_domain=sys.argv[2]
run_spe=sys.argv[3]

print(f'source domain: {source_domain}')
print(f'target domain: {target_domain}')
model_path='./snapshot/'+source_domain+'_b'+run_spe+'.pth'

if source_domain == "drive":
    dataset_mean=[0.4969, 0.2702, 0.1620]
    dataset_std=[0.3479,0.1896,0.1075]

if source_domain == "stare":
    dataset_mean=[0.5889, 0.3272, 0.1074]
    dataset_std=[0.3458,0.1844,0.1104]

if source_domain == "chase":
    dataset_mean=[0.4416, 0.1606, 0.0277]
    dataset_std=[0.3530,0.1407,0.0366]

if source_domain == "rimone": # 
    dataset_mean = [0.3383, 0.1164, 0.0465] # In use
    dataset_std = [0.1849, 0.0913, 0.0441]

if source_domain == "refuge": # 
    dataset_mean = [0.4237, 0.2414, 0.1182] # In Use
    dataset_std  = [0.1996, 0.1206, 0.0712]

if source_domain == "refuge2": # 
    dataset_mean = [0.5984, 0.4048, 0.3161] # In Use
    dataset_std  = [0.2416, 0.1871, 0.1442]

if target_domain == "drive":
    test_data_path = "/data/95/dataset/drive/test/"

if target_domain == "chase":
    test_data_path = "/data/95/dataset/chase_db1/test/"

if target_domain == "stare":
    test_data_path = "/data/95/dataset/stare/test/"

if target_domain == "rimone":
    test_data_path = "/data/95/dataset/oc/rimone/test/"

if target_domain == "refuge":
    test_data_path = "/data/95/dataset/oc/refuge/train_valid/"

if target_domain == "refuge2":
    test_data_path = "/data/95/dataset/oc/refuge/valid_test/"

save_path='./results/'

if __name__ == "__main__":
    with torch.no_grad():
        test_dataset = FundusSeg_Loader(test_data_path,0, target_domain, dataset_mean, dataset_std)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        print('Testing images: %s' %len(test_loader.dataset))
        device = torch.device('cuda')
        net = UNet(n_channels=3, n_classes=1)

        ### Bench Model ##
        #tensor = (torch.rand(1, 3, 256, 256),)
        #flops = FlopCountAnalysis(net, tensor)
        #print("FLOPs: ", flops.total())
        #print(parameter_count_table(net))

        net.to(device=device)
        print(f'Loading model {model_path}')
        net.load_state_dict(torch.load(model_path, map_location=device))

        net.eval()
        pre_stack = []
        label_stack = []

        for image, label, filename, raw_height, raw_width in test_loader:
            image = image.cuda().float()
            label = label.cuda().float()
            image = image.to(device=device, dtype=torch.float32)

            torch.cuda.synchronize()
            start = time.time()
            pred = net(image)
            torch.cuda.synchronize()
            end = time.time()
            #print('Inference time: %s seconds'%(end-start))
            pred = torch.sigmoid(pred)
            pred  = pred[:,:,:raw_height,:raw_width]  
            label = label[:,:,:raw_height,:raw_width]
            pred = pred.cpu().numpy().astype(np.double)[0][0]  
            label = label.cpu().numpy().astype(np.double)[0][0]

            pre_stack.append(pred)
            label_stack.append(label)

            pred = pred * 255
            save_filename = save_path + filename[0] + '.png'
            #cv2.imwrite(save_filename, pred)

        print('Evaluating...')
        label_stack = np.stack(label_stack, axis=0)
        pre_stack = np.stack(pre_stack, axis=0)
        label_stack = label_stack.reshape(-1)
        pre_stack = pre_stack.reshape(-1)

        if target_domain == "rimone" or target_domain == 'refuge' or target_domain == "refuge2":
            f1 = cal_f1(pre_stack, label_stack)
            print('f1score: {:.3f}'.format(f1))
        else:
            precision, sen, spec, f1, acc, roc_auc, pr_auc = perform_metrics(pre_stack, label_stack)
            #print(f'Precision: {precision} Sen: {sen} Spec:{spec} F1-score: {f1} Acc: {acc} ROC_AUC: {roc_auc} PR_AUC: {pr_auc}')
            print('f1score: {:.3f}'.format(f1))
