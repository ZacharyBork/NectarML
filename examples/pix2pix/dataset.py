from os       import PathLike
from pathlib  import Path

from nectarml import Tensor, vision, utils

class Pix2pixDataset(utils.data.Dataset):
    def __init__(self, root_directory: PathLike) -> None:
        super().__init__()
        # First get all the dataset files in our root directory
        self.root_directory = Path(root_directory)
        self.list_files     = list(self.root_directory.iterdir())
        
        # Then we define a transform stack. This is used for input and target.
        self.transforms = vision.transforms.Compose([
            vision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),
            vision.transforms.Resize(size=(286, 286), mode='bilinear'),
            vision.transforms.RandomHorizontalFlip(p=0.5),
            vision.transforms.RandomCrop(size=(256, 256))
        ])
        
    def __len__(self) -> int:
        # Dataset classes must define __len__. 
        return len(self.list_files)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        # Order is slightly different than other transforms libraries.
        # nectarml.vision.transforms are intended to work on tensors.
        # So first we load the current image file as a tensor:
        image_path   = self.list_files[index]
        image        = vision.utils.load_image(image_path)
        
        # Then we slice in half to create input and target tensors.
        width        = image.shape[-1] // 2
        input_image  = image[:, :, :, width:]
        target_image = image[:, :, :, :width]

        # Run our transforms on input and target.
        input_image, target_image = self.transforms(
            image=input_image, image2=target_image)
        
        # Squeeze the two tensors to remove the batch dimension.
        input_image  = input_image.squeeze(0)
        target_image = target_image.squeeze(0)

        # And finally, return them as a tuple.
        return input_image, target_image

