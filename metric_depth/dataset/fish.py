import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import pandas as pd
from pathlib import Path
import asyncio
from urllib.parse import urlparse

from scipy.ndimage import gaussian_filter

from dataset.transform import Resize, NormalizeImage, PrepareForNet
from synology_api.filestation import FileStation
from config import settings
from fishsense_core.image.raw_image import RawImage
from fishsense_core.image.rectified_image import RectifiedImage
from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_api_sdk.client import Client

class Fish(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        self.mode = mode
        self.size = size
        
        self.df = pd.read_csv(filelist_path)

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

        urlparsed = urlparse(settings.e4e_nas.url)
        self.filestation = FileStation(urlparsed.hostname, urlparsed.port, settings.e4e_nas.username, settings.e4e_nas.password, secure=True, cert_verify=False)

    async def __get_intrisics(self, image_id: int) -> CameraIntrinsics:
        async with Client(settings.fishsense_api.base_url, settings.fishsense_api.username, settings.fishsense_api.password) as client:
            image = await client.images.get(image_id=image_id)
            dive = await client.dives.get(dive_id=image.dive_id)
            intrinsics = await client.cameras.get_intrinsics(dive.camera_id)
            return intrinsics
    
    def __getitem__(self, item):    
        directory = Path('dataset/data/')
        img_path = Path(self.df.iloc[item]['image_path'])
        filetype = '.JPG'

        source_nas_path = f"/fishsense_data/REEF/data/{str(img_path)}"
        source_file = directory / img_path

        target_file = directory / img_path.parent / f"{img_path.stem}{filetype}"

        if not source_file.exists():
            directory.mkdir(parents=True, exist_ok=True)
            self.filestation.get_file(source_nas_path, "download", dest_path=str(target_file.parent))

        if not target_file.exists():
            image_id = int(self.df.iloc[item]['image_id'])

            image = RawImage(source_file)
            intrinsics = asyncio.run(self.__get_intrisics(image_id))
            image = RectifiedImage(image, intrinsics)

            cv2.imwrite(str(target_file), image.data)

        x = float(self.df.iloc[item]['label_x'])
        y = float(self.df.iloc[item]['label_y'])
        
        image = cv2.imread(str(directory / img_path.parent / f"{img_path.stem}{filetype}"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        H, W, _ = image.shape

        # Width of Gaussian for training heatmap
        sigma = 25

        yy = np.arange(H)[:, None]
        xx = np.arange(W)[None, :]

        # Gaussian formula
        zeros = np.exp(-((xx - x)**2 + (yy - y)**2) / (2*sigma**2))

        # Ensure annotation pixel is a unique maximum value
        zeros[int(round(y)), int(round(x))] = zeros.max() + 1e-6

        zeros /= zeros.max()
        
        sample = self.transform({'image': image, 'laser': zeros})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['laser'] = torch.from_numpy(sample['laser'])
        sample['image_path'] = str(target_file)
        sample['laser_loc'] = [x, y]
        
        return sample

    def __len__(self):
        return self.df.shape[0]