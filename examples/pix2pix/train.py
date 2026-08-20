# This file contains an example training script for a Pix2Pix-style model in
# NectarML. This file will give a practical demonstration of various concepts 
# including:
#
#    1. Initilizing models and optimizers, and loss modules.
#    2. Saving and loading model checkpoints.
#    3. Defining learning rate schedules.
#    4. Building and interacting with datasets and dataloaders.
#    5. Defining forward and backward passes for networks.
#    6. Using autocast contexts to increase performance for CUDA training.
#    7. Using gradient scalers to stabilize mixed/half precision training.
#

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
from   nectarml import nn, optim, utils

import common 
from   generator     import Generator
from   discriminator import Discriminator
from   dataset       import Pix2pixDataset

###############################################################################
# CONFIGURATION
###############################################################################

### MODEL / TRAINING SETTINGS ###

DEVICE           = 'cuda' # The device to run the training on.
LR               = 0.0002 # Base learning rate for G and D.
NUM_EPOCHS       = 100    # The number of epochs to maintain the base LR.
NUM_EPOCHS_DECAY = 100    # The number of epochs over which to decay LR to 0.
L1_LAMBDA        = 100.0  # Weighting factor for generator L1 loss

### CHECKPOINT LOADING ###

CHECKPOINT_G = '' # Sys path to G checkpoint to load, or '' for no checkpoint.
CHECKPOINT_D = '' # Sys path to D checkpoint to load, or '' for no checkpoint.

### DATALOADING SETTINGS ###

DIRECTION: Literal['AtoB', 'BtoA'] = 'AtoB' # Dataset transform direction.
BATCH_SIZE  = 1                             # Batch size for training.
NUM_WORKERS = 0                             # Training dataloader worker count.

TRAIN_SET_PATH = '' # Sys path to training images directory.
VAL_SET_PATH   = '' # Sys path to validation images directory.

### OUTPUT SETTINGS ###

OUTPUT_DIRECTORY  = ''   # Root output directory path.
ALLOW_EXISTING    = True # Whether to allow existing output directories.
MODEL_SAVE_RATE   = 10   # Rate (in epochs) at which to save model checkpoints.
EXAMPLE_SAVE_RATE = 1    # Rate (in epochs) at which to save example images.

### VISUALIZATION SETTINGS ###

UPDATE_FREQ           = 50    # Update frequency (iters) for console and web.
ENABLE_WEB_VISUALIZER = False # Enables web visualizer (images, loss graphs).
WEB_VISUALIZER_HOST   = 'http://localhost' # Host name of web viz server
WEB_VISUALIZER_PORT   = 8097               # Port number of web viz server

# To use the web visualizer, you must first start a server by running the 
# command `python -m nectarml.viz.web` from the repository root. Then in your 
# browser, navigate to the url output by the command.

###############################################################################
# LOSS MONITORING
###############################################################################

LOSSES      = { 'd_real': 0, 'd_fake': 0, 'g_gan': 0, 'g_l1': 0 }
LOSSES_PREV = LOSSES.copy()

def update_losses(
    D_real_loss: nectarml.Tensor,
    D_fake_loss: nectarml.Tensor,
    G_fake_loss: nectarml.Tensor,
    G_L1_loss:   nectarml.Tensor
) -> None:
    global LOSSES
    LOSSES['d_real'] += D_real_loss.mean().item()
    LOSSES['d_fake'] += D_fake_loss.mean().item()
    LOSSES['g_gan']  += G_fake_loss.mean().item()
    LOSSES['g_l1']   += G_L1_loss.mean().item()

def update_visualizers(
    step:      int,
    iteration: int,
    x:         nectarml.Tensor,
    y_fake:    nectarml.Tensor,
    y:         nectarml.Tensor
) -> None:
    global LOSSES, LOSSES_PREV
    for loss in LOSSES: LOSSES[loss] /= UPDATE_FREQ
    
    common.print_losses(iteration, LOSSES, LOSSES_PREV)
    if ENABLE_WEB_VISUALIZER:
        common.update_web_images(
            tensors = [x, y_fake, y], 
            title   = 'Input | Fake | Target')
        common.update_web_graph(
            step   = step, 
            losses = [LOSSES['g_gan'], LOSSES['g_l1']],
            legend = ['G_GAN', 'G_L1'],
            window = 'loss_g',
            title  = 'Generator Loss')
        common.update_web_graph(
            step   = step, 
            losses = [LOSSES['d_real'], LOSSES['d_fake']],
            legend = ['D_REAL', 'D_FAKE'],
            window = 'loss_d',
            title  = 'Discriminator Loss')
    
    LOSSES_PREV = LOSSES.copy()
    for loss in LOSSES: LOSSES[loss] = 0
        
###############################################################################
# TRAIN LOOP FUNCTION
###############################################################################

def train_fn(
    disc:         Discriminator, 
    gen:          Generator, 
    opt_disc:     optim.Optimizer,
    opt_gen:      optim.Optimizer, 
    g_scaler:     nectarml.amp.GradScaler, 
    d_scaler:     nectarml.amp.GradScaler,
    train_loader: utils.data.Dataloader, 
    L1:           nn.L1Loss, 
    BCE:          nn.BCEWithLogitsLoss,
    epoch:        int
) -> None:    
    for idx, (x, y) in enumerate(train_loader): 
        iteration = idx + 1
                
        ### DATALOADING ###

        x, y = x.to(DEVICE), y.to(DEVICE)
        
        ### GENERATOR INFERENCE ###
                        
        with nectarml.amp.autocast('cuda'): y_fake = gen(x)
        
        ### DISCRIMINATOR (FORWARD) ###
            
        with nectarml.amp.autocast('cuda'):
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            
            D_real_loss = BCE(D_real, nectarml.ones_like(D_real))
            D_fake_loss = BCE(D_fake, nectarml.zeros_like(D_fake))
            D_loss      = (D_real_loss + D_fake_loss) / 2
            
        ### DISCRIMINATOR (BACKWARD) ###

        disc.zero_grad()

        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        ### GENERATOR (FORWARD) ###
                            
        with nectarml.amp.autocast('cuda'):
            D_fake      = disc(x, y_fake)
            G_fake_loss = BCE(D_fake, nectarml.ones_like(D_fake))
            G_L1_loss   = L1(y_fake, y) * L1_LAMBDA
            G_loss      = G_fake_loss + G_L1_loss
                        
        ### GENERATOR (BACKWARD) ###

        opt_gen.zero_grad()
        
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()  
        
        ### POST-ITER ###

        if iteration != 0 and iteration % UPDATE_FREQ == 0: 
            step = (epoch-1) + idx / len(train_loader)
            update_visualizers(step, iteration, x, y_fake, y)
        else: update_losses(D_real_loss, D_fake_loss, G_fake_loss, G_L1_loss)

###############################################################################
# TRAINING LOOP FUNCTION
###############################################################################

def train() -> None:
    
    ### BUILD DIRECTORY STRUCTURE ###
    
    output_path   = common.build_output_dir(OUTPUT_DIRECTORY, ALLOW_EXISTING)
    examples_path = common.build_examples_dir(output_path, ALLOW_EXISTING)
    
    ### INITIALIZE NETWORKS ###
    
    gen  = Generator(in_channels=3).to(DEVICE)
    disc = Discriminator(in_channels=3).to(DEVICE)

    ### INITIALIZE OPTIMIZERS ###
    
    opt_gen  = optim.Adam(gen.parameters(),  lr=LR, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
    
    ### INITIALIZE GRADIENT SCALERS ###
    
    g_scaler = nectarml.amp.GradScaler()
    d_scaler = nectarml.amp.GradScaler()
    
    ### INITIALIZE LOSS MODULES ###

    BCE     = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    ### LOAD CHECKPOINTS ###
    
    start_epoch = 0

    if not CHECKPOINT_G == '':
        path_g = Path(CHECKPOINT_G).resolve()
        if not path_g.exists():
            raise FileNotFoundError(
                f'Unable to locate generator checkpoint at: '
                f'{path_g.as_posix()}')
        
        info = nn.utils.checkpoint(model=gen, optimizer=opt_gen).load(path_g)
        start_epoch = info['epoch']
    
    if not CHECKPOINT_D == '':
        path_d = Path(CHECKPOINT_D).resolve()
        if not path_d.exists():
            raise FileNotFoundError(
                f'Unable to locate discriminator checkpoint at: '
                f'{path_d.as_posix()}')
        
        nn.utils.checkpoint(model=disc, optimizer=opt_disc).load(path_d)
    
    ### INITIALIZE LR SCHEDULES ###
    
    sched_g = optim.SequentialLR(
        optimizer=opt_gen, 
        schedulers=[
            optim.ConstantLR(opt_gen, factor=1.0, total_iters=NUM_EPOCHS),
            optim.LinearLR(
                opt_gen, 
                start_factor = 1.0, 
                end_factor   = 0.0, 
                total_iters  = NUM_EPOCHS_DECAY)
        ],
        milestones=[NUM_EPOCHS], 
        last_epoch=start_epoch
    )
    
    sched_d = optim.SequentialLR(
        optimizer=opt_disc, 
        schedulers=[
            optim.ConstantLR(opt_disc, factor=1.0, total_iters=NUM_EPOCHS),
            optim.LinearLR(
                opt_disc, 
                start_factor = 1.0, 
                end_factor   = 0.0, 
                total_iters  = NUM_EPOCHS_DECAY)
        ],
        milestones=[NUM_EPOCHS], 
        last_epoch=start_epoch
    )
    
    ### INITIALIZE GRADIENT SCALERS ###
    
    g_scaler = nectarml.amp.GradScaler()
    d_scaler = nectarml.amp.GradScaler()
    
    ### INITIALIZE LOSS MODULES ###

    BCE     = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    ### BUILD DATASETS / DATALOADERS ###
    
    train_dataset = Pix2pixDataset(TRAIN_SET_PATH, DIRECTION, DEVICE)
    train_loader  = utils.data.Dataloader(
        train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    val_dataset = Pix2pixDataset(
        VAL_SET_PATH, DIRECTION, DEVICE, training=False)
    
    ### RUN TRAINING LOOP ###

    for idx in range(start_epoch, NUM_EPOCHS + NUM_EPOCHS_DECAY):
        epoch = idx + 1
        print(f'{'='*os.get_terminal_size()[0]}\nBeginning epoch {epoch}...')
        
        ### RUN EPOCH WITH TIME BENCHMARKING ###
        
        with utils.benchmark_time(f'Finished epoch {epoch}', new_line=True):
            train_fn(
                disc, gen, opt_disc, opt_gen, g_scaler, d_scaler,
                train_loader, L1_LOSS, BCE, epoch
            )

        ### STEP SCHEDULERS ###
        
        last_lr_g, last_lr_d = sched_g.get_lr(), sched_d.get_lr()

        sched_g.step(); sched_d.step()
        
        print(f'Learning Rate (G): {last_lr_g:.4f} -> {sched_g.get_lr():.4f}')
        print(f'Learning Rate (D): {last_lr_d:.4f} -> {sched_d.get_lr():.4f}')
        
        ### SAVE EXAMPLE IMAGES ###
        
        if epoch % EXAMPLE_SAVE_RATE == 0:
            common.save_examples(gen, val_dataset, examples_path, epoch)
            
        ### SAVE MODEL CHECKPOINTS ###
        
        if epoch % MODEL_SAVE_RATE == 0:
            path_g = Path(output_path, f'epoch{epoch}_netG.nml.tar')
            nn.utils.checkpoint(gen, opt_gen).save(path_g, epoch=epoch)
            
            path_d = Path(output_path, f'epoch{epoch}_netD.nml.tar')
            nn.utils.checkpoint(disc, opt_disc).save(path_d, epoch=epoch)
        
if __name__ == '__main__':
    if ENABLE_WEB_VISUALIZER: 
        common.init_web_client(
            host = WEB_VISUALIZER_HOST, 
            port = WEB_VISUALIZER_PORT)
        common.clear_web_visualizer()
    
    train()


