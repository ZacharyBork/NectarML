import os
from   typing  import Literal
from   pathlib import Path

import nectarml
from   nectarml import nn, optim, utils

from model   import ResNet, ResNet50, ResNet101, ResNet152
from dataset import ResNetDataset

###############################################################################
# CONFIGURATION
###############################################################################

### MODEL SETTINGS ###

NUM_LAYERS: Literal['50', '101', '152'] = '50'
NUM_CLASSES    = 10
IMAGE_CHANNELS = 3

DEVICE            = 'cuda'
LR                = 0.01
BATCH_SIZE        = 24
NUM_EPOCHS        = 200
AUTOCAST_ENABLED  = True

### OUTPUT SETTINGS ###

OUTPUT_DIRECTORY  = ''
ALLOW_EXISTING    = True
MODEL_SAVE_RATE   = 10

### CHECKPOINT LOADING ###

CHECKPOINT_PATH = ''

### DATASET SETTINGS ###

TRAIN_SET_PATH = ''
VAL_SET_PATH   = ''

### CONSOLE SETTINGS ###

CONSOLE_UPDATE_FREQ = 10

###############################################################################
# TRAIN LOOP FUNCTION
###############################################################################

def train_fn(
    model:        ResNet,
    opt:          optim.Optimizer, 
    scaler:       nectarml.amp.GradScaler, 
    train_loader: utils.data.Dataloader, 
    criterion:    nn.CrossEntropyLoss 
) -> None:
    for idx, (x, y) in enumerate(train_loader): 
        iteration = idx + 1
                
        ### DATALOADING ###

        x, y = x.to(DEVICE), y.to(DEVICE)
        
        ### FORWARD ###
                        
        with nectarml.amp.autocast('cuda', enabled=True):
            logits = model(x)
            loss   = criterion(logits, y)

        ### BACKWARD ###
        
        model.zero_grad()
        
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
        
        ### POST-ITER ###

        if iteration != 0 and iteration % CONSOLE_UPDATE_FREQ == 0: 
            predicted = logits.argmax(dim=1)
            correct   = (predicted == y.to(dtype=predicted.dtype))
            accuracy  = correct.float().sum().item() / len(y)
            correct   = ['x' if i else 'o' for i in correct.tolist()]
            correct   = f'[{' '.join([i for i in correct])}]'
            
            print(f'{'='*os.get_terminal_size()[0]}\n'
                  f'Iteration: {iteration}\n'
                  f'Metrics:\n'
                  f'    Predicted: {predicted.cpu().numpy()}\n'
                  f'    Actual:    {y.cpu().numpy()}\n'
                  f'    Correct:   {correct}\n'
                  f'    Accuracy:  {accuracy:.4f}\n')

###############################################################################
# VALIDATION LOOP FUNCTION
###############################################################################

def val_fn(
    model:      ResNet,
    val_loader: utils.data.Dataloader,
    epoch:      int
) -> None:
    model.eval()
    total_correct = total_samples = 0
    
    for x, y in val_loader:
        x, y      = x.to(DEVICE), y.to(DEVICE)
        logits    = model(x)
        predicted = logits.argmax(dim=1)
        
        total_correct += (predicted == y).float().sum().item()
        total_samples += len(y)

    val_accuracy = total_correct / total_samples
    print(f'Epoch {epoch} : val accuracy: {val_accuracy:.4f}')
    model.train()

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
# TRAINING LOOP FUNCTION
###############################################################################

def train() -> None:
    
    ### BUILD DIRECTORY STRUCTURE ###
    
    output_path = build_output_directory()
    
    ### INITIALIZE NETWORK ###
    
    match NUM_LAYERS:
        case '50':  factory = ResNet50
        case '101': factory = ResNet101
        case '125': factory = ResNet152
        case _: raise ValueError(f'NUM_LAYERS not valid.')
        
    model = factory(IMAGE_CHANNELS, NUM_CLASSES).to(DEVICE)

    ### INITIALIZE OPTIMIZER ###
    
    opt = optim.SGD(
        model.parameters(), 
        lr=LR, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    
    ### INITIALIZE GRADIENT SCALER ###
    
    scaler = nectarml.amp.GradScaler()
    
    ### INITIALIZE LOSS MODULE ###

    criterion = nn.CrossEntropyLoss()

    ### LOAD CHECKPOINT ###
    
    start_epoch = 0

    if not CHECKPOINT_PATH == '':
        path = Path(CHECKPOINT_PATH).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f'Unable to locate generator checkpoint at: '
                f'{path.as_posix()}')
        
        info = nn.utils.checkpoint(model=model, optimizer=opt).load(path)
        start_epoch = info['epoch']
    
    ### BUILD DATALOADER ###
    
    train_dataset = ResNetDataset(TRAIN_SET_PATH)
    train_loader  = utils.data.Dataloader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    if not VAL_SET_PATH == '': 
        val_dataset = ResNetDataset(VAL_SET_PATH)
        val_loader  = utils.data.Dataloader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    ### RUN TRAINING LOOP ###

    for idx in range(start_epoch, NUM_EPOCHS):
        epoch = idx + 1
        print(f'Beginning epoch {epoch}...')
        
        ### TRAIN MODEL ###
        
        with utils.benchmark_time(f'Finished epoch {epoch}', new_line=True):
            train_fn(model, opt, scaler, train_loader, criterion)
            
        ### VALIDATE MODEL ###
            
        if not VAL_SET_PATH == '': val_fn(model, val_loader, epoch)
        
        ### SAVE CHECKPOINTS ###
        
        if epoch % MODEL_SAVE_RATE == 0:
            checkpoint_path = Path(output_path, f'epoch{epoch}_netG.pth.tar')
            nn.utils.checkpoint(model, opt).save(checkpoint_path, epoch=epoch)

        
if __name__ == '__main__':
    train()


