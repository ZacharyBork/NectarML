from os       import PathLike
from pathlib  import Path
from typing   import Literal

from   nectarml import Tensor, vision, utils
import nectarml.vision.transforms as xforms

class Pix2pixDataset(utils.data.Dataset):
    def __init__(
        self, 
        root_directory: PathLike, 
        direction:      Literal['AtoB', 'BtoA'] = 'AtoB',
        device:         Literal['cpu',  'cuda'] = 'cpu',
        training:       bool = True
    ) -> None:
        super().__init__()
        self.training = training
        
        # First get all the dataset files in our root directory.
        self.root_directory = Path(root_directory)
        self.list_files     = list(self.root_directory.iterdir())
        self.reverse        = direction.strip().casefold() == 'btoa'
        
        # Then we define an augmentation setup for the data.
        self.transforms = xforms.Compose(
            # Many NectarML transforms can run natively on the GPU. Useful for
            # expensive transforms when high worker count isn't required.
            xforms.ToCUDA() if device == 'cuda' else xforms.NoOp(),
            
            # Add variation to input & target
            xforms.RandomHorizontalFlip(p=0.5),
            xforms.Resize(size=(286, 286), mode='bilinear'),
            xforms.RandomCrop(size=(256, 256))            
        )
        
        # And finally, define a Normalize transform to normalize both
        # input and target to [-1:1] to match tanh output range.
        self.normalize = xforms.Normalize(mean=0.5, std=0.5)

    def __len__(self) -> int:
        # Dataset classes must define __len__. 
        return len(self.list_files)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        # Order is slightly different than other transforms libraries.
        # nectarml.vision.transforms are intended to work on tensors.
        
        # So first we get our image filepath:
        image_path = self.list_files[index]
        
        # Then we load it directly as a tensor. "normalize=True" will 
        # normalize the loaded image to a saturated [0:1] range.
        image = vision.utils.load_image(image_path, normalize=True)
                
        # Then slice in half to create a/b tensors.
        width = image.shape[-1] // 2
        a     = image[:, :, :, width:]
        b     = image[:, :, :, :width]
        
        # Decide input vs. target from dataset "direction".
        input, target = (a, b) if self.reverse else (b, a)

        # Run our transforms.
        if self.training: # All transforms if training.
            input, target = self.transforms(image=input, image2=target)
            input, target = self.normalize(image=input, image2=target)
        else: # Otherwise we just normalize the data [-1:1].
            input, target = self.normalize(image=input, image2=target)
        
        # Squeeze the two tensors to remove the batch dimension.
        input, target = input.squeeze(0), target.squeeze(0)

        # And return the result as a tuple.
        return input, target

