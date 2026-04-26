import os
from   pathlib import Path
from   typing  import Literal

import nectarml
from   nectarml import nn, optim, utils
from   nectarml.viz.web import Client

from generator     import Generator
from discriminator import Discriminator
from dataset       import Pix2pixDataset

###############################################################################
# CONFIGURATION
###############################################################################

### MODEL / TRAINING SETTINGS ###

DEVICE            = 'cuda'
LR                = 0.002
NUM_EPOCHS        = 100
NUM_EPOCHS_DECAY  = 100
L1_LAMBDA         = 100.0

### CHECKPOINT LOADING ###

CHECKPOINT_G = ''
CHECKPOINT_D = ''

### DATALOADING SETTINGS ###

DIRECTION: Literal['AtoB', 'BtoA'] = 'AtoB'
BATCH_SIZE  = 1
NUM_WORKERS = 0

TRAIN_SET_PATH = ''
VAL_SET_PATH   = ''

### OUTPUT SETTINGS ###

OUTPUT_DIRECTORY  = ''
ALLOW_EXISTING    = True
MODEL_SAVE_RATE   = 10
EXAMPLE_SAVE_RATE = 1

### VISUALIZATION SETTINGS ###

UPDATE_FREQ           = 50
ENABLE_WEB_VISUALIZER = False

if ENABLE_WEB_VISUALIZER:
    VISUALIZER = Client(host='http://localhost', port=8097)

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
    BCE:          nn.BCELoss,
    epoch:        int
) -> None:
    loss_totals = { 'd_real': 0, 'd_fake': 0, 'g_gan': 0, 'g_l1': 0 }
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
        
        ### POST-ITER ###
        
        if iteration != 0 and iteration % UPDATE_FREQ == 0: 
            for loss in loss_totals: loss_totals[loss] /= UPDATE_FREQ
            print(f'{'='*os.get_terminal_size()[0]}\n'
                  f'Iteration: {iteration}\n'
                  f'Loss:\n'
                  f'    D_real: {loss_totals['d_real']:.4f}\n'
                  f'    D_fake: {loss_totals['d_fake']:.4f}\n'
                  f'    G_GAN:  {loss_totals['g_gan']:.4f}\n'
                  f'    G_L1:   {loss_totals['g_l1']:.4f}\n')
            
            if ENABLE_WEB_VISUALIZER:
                examples = [i[0].detach().cpu() for i in [x, y_fake, y]]
                VISUALIZER.images(
                    examples, size=256, normalize=True,
                    window='images', title='Input | Fake | Target')
                VISUALIZER.line(
                    X      = epoch + idx / len(train_loader),
                    Y      = [_ for _ in loss_totals.values()],
                    legend = [_ for _ in loss_totals],
                    window = 'loss', 
                    title  = 'Loss', 
                    v_axis_label='Value', h_axis_label='Epoch'
                )
            
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
    
    path.mkdir(exist_ok=ALLOW_EXISTING)
    return path

def build_examples_directory(output_path: Path) -> Path:
    '''Builds subdirectory for example output images.
    
    Args:
        output_path : The path to the root output directory.
    
    Returns:
        Path : The path to the newly created example directory.
    '''
    examples_directory = Path(output_path, 'examples').resolve()
    examples_directory.mkdir(exist_ok=ALLOW_EXISTING)
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
    
    sched_d = sched_g.clone(new_optimizer=opt_disc)
    
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
            save_examples(gen, val_dataset, examples_path, epoch)
            
        ### SAVE MODEL CHECKPOINTS ###
        
        if epoch % MODEL_SAVE_RATE == 0:
            path_g = Path(output_path, f'epoch{epoch}_netG.pth.tar')
            nn.utils.checkpoint(gen, opt_gen).save(path_g, epoch=epoch)
            
            path_d = Path(output_path, f'epoch{epoch}_netD.pth.tar')
            nn.utils.checkpoint(disc, opt_disc).save(path_d, epoch=epoch)
        
if __name__ == '__main__':
    if ENABLE_WEB_VISUALIZER: VISUALIZER.clear()
    train()


