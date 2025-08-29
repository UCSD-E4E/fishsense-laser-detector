import argparse
import logging
import os
import pprint
import random
import time

import cv2
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

from dataset.fish import Fish
# from dataset.hypersim import Hypersim
# from dataset.kitti import KITTI
# from dataset.vkitti2 import VKITTI2
from depth_anything_v2.dpt import DepthAnythingV2
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log


parser = argparse.ArgumentParser(description='Depth Anything V2 for Laser Prediction')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='fish', choices=['hypersim', 'vkitti', 'fish'])
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--bs', default=1, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from', default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', type=str)
parser.add_argument('--save-path', default='checkpoints',type=str)

def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    warnings.simplefilter('ignore', np.exceptions.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (args.img_size, args.img_size)

    trainset = Fish('dataset/splits/fish/train.txt', 'train', size=size)

    # trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True)
    
    valset = Fish('dataset/splits/fish/val.txt', 'val', size=size)

    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    if args.pretrained_from:
        model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
    
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    total_iters = args.epochs * len(trainloader)
    
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    # previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100, 'distance': 5020}

    for epoch in range(args.epochs):
        start_train = time.perf_counter()
        model.train()
        total_loss = 0
        
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            
            img, laser = sample['image'].cuda(), sample['laser'].cuda()

            if random.random() < 0.5:
                img = img.flip(-1)  # horizontal flip
                laser = laser.flip(-1)

            if random.random() < 0.5:
                img = img.flip(-2)  # vertical flip
                laser = laser.flip(-2)
            
            pred = model(img)
            loss = criterion(pred, laser)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0
        end_train = time.perf_counter()
        print('Training time:', end_train-start_train)

        start_eval = time.perf_counter()
        model.eval()
        
        results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
                   'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
        
        # results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
        #            'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
        #            'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda(),
        #            'distance': torch.tensor([0.0]).cuda()}

        nsamples = torch.tensor([0.0]).cuda()

        kernel_size = 100
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        
        for i, sample in enumerate(valloader):

            # Stop after 100 images
            # if i > 100:
            #     break
            
            img, laser = sample['image'].cuda().float(), sample['laser'].cuda()[0]
            
            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(pred[:, None], laser.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
            valid_mask = (laser >= args.min_depth) & (laser <= args.max_depth)
            
            if valid_mask.sum() < 10:
                continue

            cur_results = eval_depth(pred[valid_mask], laser[valid_mask])

            # Eval Euclidean dist from predection to laser annotation
            # print(sample['image_path'])
            # raw_image = cv2.imread(sample['image_path'][0])
            # raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

            # pred_full = model.infer_image(raw_image, input_size=518)
            # pred_full = (pred_full - pred_full.min()) / (pred_full.max() - pred_full.min() + 1e-8)
            # pred_full = (pred_full * 255).astype(np.uint8)
            # # mean vs median vs raw laser prediction
            # # response = cv2.filter2D(pred_full.astype(np.float32), ddepth=-1, kernel=kernel)
            # response = cv2.medianBlur(pred_full, 41)
            # # response = pred_full.astype(np.float32)
            # _, _, _, max_loc = cv2.minMaxLoc(response)
            # pred_x, pred_y = max_loc

            # # max_val = response.max()
            # # max_locs = np.argwhere(response == max_val)
            # # if len(max_locs) > 1:
            # #     print('Maximums:', len(max_locs))
            # # pred_x, pred_y = max_locs[0]

            # targ_x, targ_y = sample['laser_loc']

            # distance = torch.tensor(np.sqrt((pred_x - targ_x) ** 2 + (pred_y - targ_y) ** 2)).cuda()
            # cur_results['distance'] = distance
            # print('Distance:', distance, 'From:', pred_x, pred_y, targ_x, targ_y)
            
            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1
        
        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'median.pth'))
    # print('Distance:', results['distance'], 'Samples:', len(valloader))
    end_eval = time.perf_counter()
    print('Eval time:', end_eval-start_eval)


if __name__ == '__main__':
    main()