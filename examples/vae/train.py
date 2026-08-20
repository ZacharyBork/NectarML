###############################################################################
# IMPORTS
###############################################################################

import sys
import os
from   pathlib import Path
from   typing  import Literal

# Add examples root to access common utilities
sys.path.insert(0, Path(__file__).parent.joinpath('..').as_posix())

import nectarml
import nectarml.nn.functional as F
from   nectarml import nn, optim, utils

import common 
from   model   import VAE
from   dataset import VAEDataset

###############################################################################
# CONFIGURATION
###############################################################################

### MODEL / TRAINING SETTINGS ###

DEVICE           = 'cuda' # The device to run the training on.
LR               = 0.001  # Initial learning rate
NUM_EPOCHS       = 100    # Total number of training epochs.

IN_CHANNELS      = 3      # Number of channels in input images.
CROP_SIZE        = 178    # Input images will be center-cropped to this size.
INPUT_SIZE       = 128    # And then scaled to this size.
LATENT_DIM       = 192    # Feature width at bottleneck.
BETA             = 0.2    # Weighting term for KL divergence loss.
ANNEALING_EPOCHS = 20     # Epochs over which to anneal the beta value.
RECON_TYPE: Literal['l1', 'l2', 'bce'] = 'l1' # Reconstruction loss type.

### CHECKPOINT LOADING ###

CHECKPOINT = '' # Sys path to checkpoint to load, or '' for no checkpoint.

### DATALOADING SETTINGS ###

BATCH_SIZE  = 31 # Batch size for training.
NUM_WORKERS = 0  # Training dataloader worker count.

TRAIN_SET_PATH = '' # Sys path to training images directory.
VAL_SET_PATH   = '' # Sys path to validation images directory.

### OUTPUT SETTINGS ###

OUTPUT_DIRECTORY  = ''   # Root output directory path.
ALLOW_EXISTING    = True # Whether to allow existing output directories.
MODEL_SAVE_RATE   = 1    # Rate (in epochs) at which to save model checkpoints.
EXAMPLE_SAVE_RATE = 1    # Rate (in epochs) at which to save example images.

### VISUALIZATION SETTINGS ###

UPDATE_FREQ           = 100   # Update frequency (iters) for console and web.
ENABLE_WEB_VISUALIZER = False # Enables web visualizer (images, loss graphs).
WEB_VISUALIZER_HOST   = 'http://localhost' # Host name of web viz server
WEB_VISUALIZER_PORT   = 8097               # Port number of web viz server

# To use the web visualizer, you must first start a server by running the 
# command `python -m nectarml.viz.web` from the repository root. Then in your 
# browser, navigate to the url output by the command.

###############################################################################
# LOSS
###############################################################################

LOSSES      = { 'total': 0, 'recon': 0, 'kl': 0 }
LOSSES_PREV = LOSSES.copy()

def kl_divergence_loss(
    mu:     nectarml.Tensor, 
    logvar: nectarml.Tensor
) -> nectarml.Tensor:
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum())

def update_losses(
    total_loss: nectarml.Tensor,
    recon_loss: nectarml.Tensor,
    kl_loss:    nectarml.Tensor
) -> None:
    global LOSSES
    LOSSES['total'] += total_loss.mean().item()
    LOSSES['recon'] += recon_loss.mean().item()
    LOSSES['kl']    += kl_loss.mean().item()

def reset_loss_trackers() -> None:
    global LOSSES, LOSSES_PREV
    LOSSES      = { 'total': 0, 'recon': 0, 'kl': 0 }
    LOSSES_PREV = LOSSES.copy()

def update_visualizers(
    step:      int,
    iteration: int,
    x:         nectarml.Tensor,
    x_hat:     nectarml.Tensor
) -> None:
    global LOSSES, LOSSES_PREV
    for loss in LOSSES: LOSSES[loss] /= UPDATE_FREQ
    
    common.print_losses(iteration, LOSSES, LOSSES_PREV, precision=8)
    if ENABLE_WEB_VISUALIZER:
        common.update_web_images(
            tensors = [x, x_hat], 
            title   = 'x | x̂')
        common.update_web_graph(
            step   = step, 
            losses = [LOSSES['recon']],
            legend = ['Recon'],
            window = 'recon_loss',
            title  = 'Reconstruction')
        common.update_web_graph(
            step   = step, 
            losses = [LOSSES['kl']],
            legend = ['KL'],
            window = 'kl_loss',
            title  = 'KL Divergence')
    
    LOSSES_PREV = LOSSES.copy()
    for loss in LOSSES: LOSSES[loss] = 0 
    
def save_examples(
    model:       nectarml.nn.Module, 
    dataset:     nectarml.utils.data.Dataset,
    output_path: Path,
    epoch:       int,
) -> None:
    '''Saves input, target, and inference example images to disk.

    Args:
        model       : The model to use for inference.
        dataset     : The dataset to draw the test data from.
        output_path : The system path to the directory to save the images to.
        epoch       : The epoch when the example images are being saved. Used
                      as a tag in the output file names.
        x_tag       : The tag to add to the x image file name.
        y_tag       : The tag to add to the y image file name.
    '''
    model.eval()

    with nectarml.no_grad():
        idx = nectarml.random.RNG.randint(0, len(dataset)-1)
        x   = dataset[idx].unsqueeze(0).to(DEVICE)
        y   = model(x)[0]
        for i in [(x, 'input'), (y, 'output')]:
            image_path = Path(output_path, f'epoch{epoch}_{i[1]}.jpg')
            nectarml.vision.utils.save_image(i[0], image_path, normalize=True)
        
    model.train()
    
###############################################################################
# TRAIN LOOP FUNCTION
###############################################################################

def train_fn(
    model:      VAE, 
    optimizer:  optim.Optimizer,
    dataloader: utils.data.Dataloader,
    loss_fn:    nn.Module,
    epoch:      int
) -> None:
    loss_total = 0.0 # Track total loss for lr scheduler
    for idx, x in enumerate(dataloader): 
        iteration = idx + 1 # Compute time-based values
        step      = (epoch-1) + idx / len(dataloader)
        
        ### MOVE TENSOR TO DEVICE ###

        x = x.to(DEVICE)
        
        ### APPLY BETA ANNEALING ###
        
        if ANNEALING_EPOCHS > 0:
              beta = min(BETA, BETA * (step / ANNEALING_EPOCHS))
        else: beta = BETA
        
        ### FORWARD PASS ###

        x_hat, mu, logvar = model(x)
                
        recon = loss_fn(x_hat, x) / x.shape[0]
        kl    = beta * kl_divergence_loss(mu, logvar) / x.shape[0]
        loss  = recon + kl
                
        ### BACKWARD PASS ###

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### POST-ITER ###

        loss_total += loss.mean().item()
        if iteration != 0 and iteration % UPDATE_FREQ == 0: 
              update_visualizers(step, iteration, x, x_hat)
        else: update_losses(loss, recon, kl)
    
    reset_loss_trackers()
    return loss_total
                                
###############################################################################
# TRAINING LOOP FUNCTION
###############################################################################

def train() -> None:
    

    ### BUILD DIRECTORY STRUCTURE ###
    
    output_path   = common.build_output_dir(OUTPUT_DIRECTORY, ALLOW_EXISTING)
    examples_path = common.build_examples_dir(output_path, ALLOW_EXISTING)
    
    ### BUILD DATASETS / DATALOADERS ###
    
    train_dataset = VAEDataset(TRAIN_SET_PATH, CROP_SIZE, INPUT_SIZE, DEVICE)
    train_loader  = utils.data.Dataloader(
        train_dataset, BATCH_SIZE, shuffle=True, drop_last=True, 
        num_workers=NUM_WORKERS)
    
    val_dataset = VAEDataset(
        VAL_SET_PATH, CROP_SIZE, INPUT_SIZE, DEVICE, training=False)

    ### INITIALIZE NETWORK ###

    # First, initialize VAE and move to device.
    model = VAE(
        in_channels  = IN_CHANNELS, 
        latent_dim   = LATENT_DIM,
        spatial_size = INPUT_SIZE
    ).to(DEVICE)
    
    # Run dummy iteration to compute LazyLinear input channels.
    model(next(iter(train_loader))[:1].to(DEVICE))
    
    # Then init weights, now that LazyLinears have weight/bias parameters.
    model.apply(model._init_weights)
        
    ### INITIALIZE OPTIMIZER ###
    
    optimizer = optim.Adam(model.parameters(), lr=LR)    
    
    ### LOAD CHECKPOINT ###
    
    start_epoch = 0

    if not CHECKPOINT == '':
        checkpoint_path = Path(CHECKPOINT).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f'Unable to locate generator checkpoint at: '
                f'{checkpoint_path.as_posix()}')
        
        checkpoint  = nn.utils.checkpoint(model=model, optimizer=optimizer)
        info        = checkpoint.load(checkpoint_path)
        start_epoch = info['epoch']
    
    ### INITIALIZE LR SCHEDULEr ###
    
    scheduler = optim.ReduceLROnPlateau(
        optimizer=optimizer, patience=5, factor=0.5
    )
    
    ### INIT LOSS FUNCTIONS ###
    
    match RECON_TYPE:
        case 'l1':  loss_fn = nn.L1Loss (reduction='sum')
        case 'l2':  loss_fn = nn.L2Loss (reduction='sum')
        case 'bce': loss_fn = nn.BCELoss(reduction='sum')

    ### RUN TRAINING LOOP ###

    for idx in range(start_epoch, NUM_EPOCHS):
        epoch = idx + 1
        print(f'{'='*os.get_terminal_size()[0]}\nBeginning epoch {epoch}...')
        
        ### RUN EPOCH WITH TIME BENCHMARKING ###
        
        with utils.benchmark_time(f'Finished epoch {epoch}', new_line=True):
            loss = train_fn(model, optimizer, train_loader, loss_fn, epoch)

        ### STEP SCHEDULERS ###
        
        last_lr = scheduler.lr
        scheduler.step(loss)
        print(f'Learning Rate: {last_lr:.6f} -> {scheduler.lr:.6f}')
        
        ### SAVE EXAMPLE IMAGES ###
        
        if epoch % EXAMPLE_SAVE_RATE == 0:
            save_examples(model, val_dataset, examples_path, epoch)
            
        ### SAVE MODEL CHECKPOINTS ###
        
        if epoch % MODEL_SAVE_RATE == 0:
            checkpoint_path = Path(output_path, f'epoch{epoch}_netG.nml.tar')
            nn.utils.checkpoint(model, optimizer).save(checkpoint_path, epoch)
            
if __name__ == '__main__':
    if ENABLE_WEB_VISUALIZER: 
        common.init_web_client(
            host = WEB_VISUALIZER_HOST, 
            port = WEB_VISUALIZER_PORT)
        common.clear_web_visualizer()
    
    train()


