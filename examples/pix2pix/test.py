from pathlib import Path
from typing  import Literal

import nectarml
from   nectarml.vision import utils

from generator import Generator
from dataset   import Pix2pixDataset

###############################################################################
# CONFIGURATION
###############################################################################

DEVICE           = 'cuda' # Device to run the model test on: ["cpu", "cuda"]
CHECKPOINT_PATH  = ''     # System path to generator checkpoint file

DIRECTION: Literal['AtoB', 'BtoA'] = 'AtoB' # Dataset direction
INPUT_DIRECTORY  = ''                       # Path to input image directory
OUTPUT_DIRECTORY = ''                       # Path to output directory

###############################################################################
# TEST SCRIPT
###############################################################################

if __name__ == '__main__':
    
    ### VALIDATE INPUT & OUTPUT DIRECTORIES ###
    
    INPUT_DIRECTORY = Path(INPUT_DIRECTORY)
    assert INPUT_DIRECTORY.exists(), (
        f'Unable to locate input directory at path: '
        f'{INPUT_DIRECTORY.as_posix()}')
    
    OUTPUT_DIRECTORY = Path(OUTPUT_DIRECTORY)
    assert OUTPUT_DIRECTORY.exists(), (
        f'Unable to locate input directory at path: '
        f'{INPUT_DIRECTORY.as_posix()}')
    
    ### INITIALIZE GENERATOR ###
    
    generator = Generator(in_channels=3, features=64).to(DEVICE)
    generator.eval()
    
    ### LOAD CHECKPOINT ###
    
    CHECKPOINT_PATH = Path(CHECKPOINT_PATH)
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f'Unable to locate checkpoint file at: '
            f'{CHECKPOINT_PATH.as_posix()}')
    
    nectarml.nn.utils.checkpoint(model=generator).load(CHECKPOINT_PATH)

    ### INITIALIZE DATASET / DATALOADER ###
    
    dataset    = Pix2pixDataset(INPUT_DIRECTORY, DIRECTION, DEVICE, False)
    dataloader = nectarml.utils.data.Dataloader(dataset)
    
    ### ITERATE DATALOADER ###
    
    with nectarml.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            
            ### CAST TENSORS, RUN INFERENCE ###
            
            x, y   = x.to(DEVICE), y.to(DEVICE)
            y_fake = generator(x)
            
            ### SAVE RESULTING IMAGES ###
            
            for item in [(x, 'A_real'), (y, 'B_real'), (y_fake, 'B_fake')]:
                path = Path(OUTPUT_DIRECTORY, f'example{idx+1}_{item[1]}.jpg')
                utils.save_image(item[0], path, normalize=True)


