import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from depth_anything_v2.dpt import DepthAnythingV2
from dataset.fish import Fish
import time

parser = argparse.ArgumentParser(description='Depth Anything V2 for Laser Prediction')

parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--input-size', type=int, default=518)
parser.add_argument('--max-depth', type=float, default=20)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--bs', default=20, type=int)

def main():
    start_time = time.perf_counter()
    args = parser.parse_args()
    size = (args.input_size, args.input_size)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    valset = Fish('dataset/splits/fish/val.txt', 'val', size=size)

    valloader = DataLoader(valset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True)

    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    checkpoint = torch.load(args.load_from, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(DEVICE).eval()

    distances = []

    kernel = 21
    print('Kernel size:', kernel)
    print('Kernel type: median')
    
    for i, sample in enumerate(valloader):

        print('Processing batch:', i)

        # Stop after 10 batches
        if i > 10:
            break
        
        imgs = sample['image'].to(DEVICE).float()
        lasers = sample['laser'].to(DEVICE)
        laser_locs = sample['laser_loc']
        
        with torch.no_grad():
            preds = model(imgs)
            preds = F.interpolate(preds[:, None], lasers.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
        
        for j in range(imgs.size(0)):

            laser = lasers[j]
            valid_mask = (laser >= args.min_depth) & (laser <= args.max_depth)
            
            if valid_mask.sum() < 10:
                continue

            # print(sample['image_path'])
            raw_image = cv2.imread(sample['image_path'][j])
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

            pred_full = model.infer_image(raw_image, input_size=518)
            pred_full = (pred_full - pred_full.min()) / (pred_full.max() - pred_full.min() + 1e-8)
            pred_full = (pred_full * 255).astype(np.uint8)
            # Kernels for experimentation
            # response = cv2.filter2D(pred_full.astype(np.float32), ddepth=-1, kernel=kernel)
            response = cv2.medianBlur(pred_full, kernel)
            # response = pred_full.astype(np.float32)
            _, _, _, max_loc = cv2.minMaxLoc(response)
            pred_x, pred_y = max_loc

            # max_val = response.max()
            # max_locs = np.argwhere(response == max_val)
            # if len(max_locs) > 1:
            #     print('Maximums:', len(max_locs))
            # pred_x, pred_y = max_locs[0]

            targ_x = laser_locs[0][j].item()
            targ_y = laser_locs[1][j].item()

            distance = np.sqrt((pred_x - targ_x) ** 2 + (pred_y - targ_y) ** 2)
            distances.append(distance)
            # print('Distance:', distance, 'From:', pred_x, pred_y, targ_x, targ_y)
    print('Mean distance', np.mean(distances))
    print('Median distance', np.median(distances))
    print('Min distance', np.min(distances))
    print('Max distance', np.max(distances))
    elapsed = time.perf_counter() - start_time
    print(f"Total time: {elapsed:.3f} seconds")

if __name__ == '__main__':
    main()