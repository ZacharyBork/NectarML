import os
from   pathlib import Path

import nectarml
from   nectarml import nn, optim, utils

from generator     import Generator
from discriminator import Discriminator
from dataset       import Pix2pixDataset

###############################################################################
# CONFIGURATION
###############################################################################

DEVICE            = 'cuda'
LR                = 0.0002
BATCH_SIZE        = 1
NUM_EPOCHS        = 200
MODEL_SAVE_RATE   = 10
EXAMPLE_SAVE_RATE = 1
L1_LAMBDA         = 100.0
AUTOCAST_ENABLED  = True
GRADIENT_SCALING  = True
SYNC_CUDA         = True

OUTPUT_DIRECTORY = ''

CHECKPOINT_G = ''
CHECKPOINT_D = ''

TRAIN_SET_PATH = ''
VAL_SET_PATH   = ''
TEST_SET_PATH  = ''

CONSOLE_UPDATE_FREQ = 5
CONSOLE_WIDTH       = os.get_terminal_size()[0]

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
    L1_LOSS:      nn.L1Loss, 
    BCE:          nn.BCELoss
) -> None:
    loss_totals = { 'd_real': 0, 'd_fake': 0, 'g_gan': 0, 'g_l1': 0 }
    for idx, (x, y) in enumerate(train_loader): 
        iteration = idx + 1
                
        ### DATALOADING ###

        x: nectarml.Tensor = x.to(DEVICE)
        y: nectarml.Tensor = y.to(DEVICE)
                                            
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
        d_scaler.unscale_(opt_disc)
        d_scaler.step(opt_disc)
        d_scaler.update()

        ### GENERATOR (FORWARD) ###
                            
        with nectarml.amp.autocast('cuda'):
            D_fake      = disc(x, y_fake)
            G_fake_loss = BCE(D_fake, nectarml.ones_like(D_fake))
            L1          = L1_LOSS(y_fake, y) * L1_LAMBDA
            G_loss      = G_fake_loss + L1

        ### GENERATOR (BACKWARD) ###

        opt_gen.zero_grad()
        
        g_scaler.scale(G_loss).backward()
        g_scaler.unscale_(opt_gen)
        g_scaler.step(opt_gen)
        g_scaler.update()         

        # ### POST-ITER ###

        if iteration != 0 and iteration % CONSOLE_UPDATE_FREQ == 0: 
            for loss in loss_totals:
                loss_totals[loss] /= CONSOLE_UPDATE_FREQ
                loss_totals[loss] = round(loss_totals[loss], 3)
            print(f'{'='*CONSOLE_WIDTH}\n'
                  f'Iteration: {iteration}\n'
                  f'Loss:\n'
                  f'    D_real: {loss_totals['d_real']}\n'
                  f'    D_fake: {loss_totals['d_fake']}\n'
                  f'    G_GAN:  {loss_totals['g_gan']}\n'
                  f'    G_L1:   {loss_totals['g_l1']}\n')
            for loss in loss_totals: loss_totals[loss] = 0
        else:
            loss_totals['d_real'] += D_real_loss.mean().item()
            loss_totals['d_fake'] += D_fake_loss.mean().item()
            loss_totals['g_gan']  += G_fake_loss.mean().item()
            loss_totals['g_l1']   += L1.mean().item()

###############################################################################
# OUTPUT STRUCTURE
###############################################################################

def build_output_directory() -> Path:
    '''Builds an directory for training outputs (checkpoints, examples).
    
    Returns:
        Path : The path to the newly created output directory.
    '''
    assert OUTPUT_DIRECTORY != '', \
        'Please set OUTPUT_DIRECTORY to begin training.'
    path = Path(OUTPUT_DIRECTORY).resolve()
    if not path.parent.exists():
        raise FileNotFoundError(
            f'Unable to build output directory at path: {path.as_posix()}\n'
            f'Parent directory does not exist.')
    if path.exists():        
        raise FileExistsError(
            f'Output directory already exists at {path.as_posix()}')
    
    path.mkdir()
    return path

def build_examples_directory(output_path: Path) -> Path:
    '''Builds subdirectory for example output images.
    
    Args:
        output_path : The path to the root output directory.
    
    Returns:
        Path : The path to the newly created example directory.
    '''
    examples_directory = Path(output_path, 'examples').resolve()
    examples_directory.mkdir()
    return examples_directory

###############################################################################
# UTILITIES
###############################################################################

def save_examples(
    gen:         Generator, 
    dataset:     utils.data.Dataset,
    output_path: Path,
    epoch:       int
) -> None:
    gen.eval()

    idx    = nectarml.random.RNG.randint(0, len(dataset)-1)
    x, y   = dataset[idx]
    x, y   = x.unsqueeze(0).to(DEVICE), y.unsqueeze(0).to(DEVICE)
    y_fake = gen(x)
    
    for item in [(x, 'A_real'), (y, 'B_real'), (y_fake, 'B_fake')]:
        image_path = Path(output_path, f'epoch{epoch}_{item[1]}.jpg')
        nectarml.vision.utils.save_image(item[0], image_path, normalize=True)
    
    gen.train()

###############################################################################
# TRAINING LOOP FUNCTION
###############################################################################

def train() -> None:
    
    ### BUILD DIRECTORY STRUCTURE ###
    
    output_path   = build_output_directory()
    examples_path = build_examples_directory(output_path)
    
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
        
        info = utils.load_checkpoint(path_g, model=gen, optimizer=opt_gen)
        start_epoch = info['epoch']
    
    if not CHECKPOINT_D == '':
        path_g = Path(CHECKPOINT_D).resolve()
        if not path_g.exists():
            raise FileNotFoundError(
                f'Unable to locate discriminator checkpoint at: '
                f'{path_g.as_posix()}')
        
        utils.load_checkpoint(path_g, model=disc, optimizer=opt_disc)
    
    ### BUILD DATASETS / DATALOADERS ###
    
    train_dataset = Pix2pixDataset(TRAIN_SET_PATH)
    train_loader  = utils.data.Dataloader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset   = Pix2pixDataset(VAL_SET_PATH)
    
    ### RUN TRAINING LOOP ###

    for idx in range(start_epoch, NUM_EPOCHS):
        epoch = idx + 1
        print(f'Beginning epoch {epoch}...')
        
        with utils.benchmark_time(f'Finished epoch {epoch}', new_line=True):
            train_fn(
                disc, gen, 
                opt_disc, opt_gen, 
                g_scaler, d_scaler,
                train_loader, 
                L1_LOSS, BCE
            )
        
        ### SAVE EXAMPLE IMAGES ###
        
        if epoch % EXAMPLE_SAVE_RATE == 0:
            save_examples(gen, val_dataset, examples_path, epoch)
            
        ### SAVE MODEL CHECKPOINTS ###
        
        if epoch % MODEL_SAVE_RATE == 0:
            path_g = Path(output_path, f'epoch{epoch}_netG.pth.tar')
            utils.save_checkpoint(path_g, gen, opt_gen, epoch=epoch)
            
            path_d = Path(output_path, f'epoch{epoch}_netD.pth.tar')
            utils.save_checkpoint(path_d, disc, opt_disc, epoch=epoch)
        
if __name__ == '__main__':
    train()


