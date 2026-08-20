from os      import PathLike
from pathlib import Path
from typing  import Literal

import nectarml
from   nectarml.vision import utils, transforms as T

class VAEDataset(nectarml.utils.data.Dataset):
    def __init__(
        self, 
        root_directory: PathLike, 
        crop_size:      int = 178,
        spatial_size:   int = 128,
        device:         Literal['cpu', 'cuda'] = 'cpu',
        training:       bool = True
    ) -> None:
        super().__init__()
        
        # First, get all files in root directory
        self.list_files = list(Path(root_directory).iterdir())
        
        # The we build a transform stack for the data
        self.transforms = T.Compose(
            # Many NectarML transforms can run natively on the GPU. Useful for
            # expensive transforms when high worker count isn't required.
            T.ToCUDA() if device == 'cuda' else T.NoOp(),
            
            # Then just center crop to our crop size, and resize to desired
            # spatial size for the model.
            T.CenterCrop(size=crop_size),
            T.Resize(size=spatial_size, mode='bilinear')
        )
        
        # If training, add some light variation with random H-flip
        if training: self.transforms.append(T.RandomHorizontalFlip(p=0.5))
        self.transforms.append(T.Normalize(mean=0.0, std=1.0))

    def __len__(self) -> int:
        # Dataset classes must define a __len__() method. 
        # Here we just have it return the number of input files.
        return len(self.list_files)
    
    def __getitem__(
        self, 
        index: int
    ) -> tuple[nectarml.Tensor, nectarml.Tensor]:
        # Get image path at index
        image_path = self.list_files[index]
        
        # Load image, normalize [0:1]
        image = utils.load_image(image_path, normalize=True)
        
        # Apply transformations
        image = self.transforms(image=image)
        
        # Squeeze to remove batch dim and return
        return image.squeeze(0)

