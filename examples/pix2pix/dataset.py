from os import PathLike
from pathlib import Path

import numpy as np
from PIL import Image

from nectarml import Tensor
from nectarml.utils.data import Dataset
from nectarml.vision import transforms

class Pix2pixDataset(Dataset):
    def __init__(
        self,
        root_directory: PathLike
    ) -> None:
        super().__init__()
        self.root_directory = Path(root_directory)
        self.list_files = list(self.root_directory.iterdir())
        
        self.both_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(286, 286), mode='bilinear'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=(256, 256)),
        ])
        
        self.transform_only_input = transforms.Compose([
            transforms.ColorJitter(p=0.5),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.transform_only_target = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self) -> int:
        return len(self.list_files)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path = self.list_files[index]
        image = np.array(Image.open(image_path))
        width = image.shape[1] // 2
        
        input_image  = image[:, :width, :]
        target_image = image[:, width:, :]
                        
        input_image, target_image = self.both_transforms(
            image=input_image, image2=target_image)
        
        input_image  = self.transform_only_input(image=input_image)
        target_image = self.transform_only_target(image=target_image)
                
        return input_image, target_image

