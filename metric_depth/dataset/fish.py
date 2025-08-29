import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from scipy.ndimage import gaussian_filter

from dataset.transform import Resize, NormalizeImage, PrepareForNet


class Fish(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        self.mode = mode
        self.size = size
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()

        # # limit to 200 images
        # random.seed(42)
        # self.filelist = random.sample(self.filelist, 200)
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def __getitem__(self, item):
        img_path, x, y, calibration = self.filelist[item].split(',')
    
        directory = 'dataset/data/'
        filetype = '.JPG'

        x = int(x)
        y = int(y)
        
        image = cv2.imread(directory+img_path+filetype)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        H, W, _ = image.shape

        # Width of Gaussian for training heatmap
        sigma = 25

        yy = np.arange(H)[:, None]
        xx = np.arange(W)[None, :]

        # Gaussian formula
        zeros = np.exp(-((xx - x)**2 + (yy - y)**2) / (2*sigma**2))

        # Ensure annotation pixel is a unique maximum value
        zeros[y, x] = zeros.max() + 1e-6

        zeros /= zeros.max()
        
        sample = self.transform({'image': image, 'laser': zeros})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['laser'] = torch.from_numpy(sample['laser'])
        sample['image_path'] = directory+img_path+filetype
        sample['laser_loc'] = [x, y]
        
        return sample

    def __len__(self):
        return len(self.filelist)