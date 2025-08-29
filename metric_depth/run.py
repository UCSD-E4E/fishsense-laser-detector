import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import time

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--save-float', dest='save_float', action='store_true', help='save the model depth output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--overlay', dest='overlay', action='store_true', help='overlay prediction onto RGB')
    parser.add_argument('--laser', dest='laser', action='store_true', help='predict laser location')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    checkpoint = torch.load(args.load_from, map_location='cpu')
    depth_anything.load_state_dict(checkpoint['model'])
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        start_time = time.perf_counter()
        depth = depth_anything.infer_image(raw_image, args.input_size)
        elapsed = time.perf_counter() - start_time
        print(f"Inference time: {elapsed:.3f} seconds")
        
        if args.save_numpy:
            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
            np.save(output_path, depth)

        if args.save_float:
            # Save 32-bit float depth map (in meters) as tiff
            float_out_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_depth_float32.tiff')
            depth_to_save = depth.astype(np.float32)
            cv2.imwrite(float_out_path, depth_to_save)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        inverse = 1.0 - depth
        inverse_norm = (inverse* 255).astype(np.uint8)
        
        if args.overlay:
            depth_overlay = (depth*255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(depth_overlay, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(raw_image, 0.7, heatmap_color, 0.3, 0)
            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_overlay.png')
            cv2.imwrite(output_path, overlay)

        if args.laser:
            kernel_size = 50
            laser_overlay = (depth*255).astype(np.uint8)
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
            response = cv2.filter2D(laser_overlay.astype(np.float32), ddepth=-1, kernel=kernel)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(response)
            print('Max location at:', max_loc[0], max_loc[1])
            cv2.circle(raw_image, max_loc, radius=10, color=(0, 0, 255), thickness=2)
            text = f"({max_loc[0]}, {max_loc[1]})"
            cv2.putText(raw_image, text, (max_loc[0] + 15, max_loc[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        if args.grayscale:
            inverse_norm = cv2.cvtColor(inverse_norm, cv2.COLOR_GRAY2BGR)
        else:
            inverse_norm = (cmap(inverse_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.pred_only:
            cv2.imwrite(output_path, inverse_norm)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, inverse_norm])
            
            cv2.imwrite(output_path, combined_result)