
import nectarml
import nectarml.nn as nn
import nectarml.optim as optim
from nectarml.utils.data import Dataloader

from generator import Generator
from discriminator import Discriminator
from dataset import Pix2pixDataset

DEVICE            = 'cuda'
LR                = 0.0002
BATCH_SIZE        = 1
NUM_EPOCHS        = 200
MODEL_SAVE_RATE   = 10
EXAMPLE_SAVE_RATE = 5
L1_LAMBDA         = 100.0

TRAIN_SET_PATH = ''
VAL_SET_PATH   = ''
TEST_SET_PATH  = ''

def train_fn(
    disc, gen, 
    train_loader, 
    opt_disc, opt_gen, 
    L1_LOSS, BCE, 
    g_scaler, d_scaler
):
    for idx, (x, y) in enumerate(train_loader): 
        
        ### DATALOADING && INFERENCE ###
        
        x, y = x[0].to(DEVICE), y[0].to(DEVICE)
        y_fake = gen(x)
        
        ### DISCRIMINATOR (FORWARD) ###

        D_real = disc(x, y)
        D_fake = disc(x, y_fake.detach())
        
        D_real_loss = BCE(D_real, nectarml.ones_like(D_real))
        D_fake_loss = BCE(D_fake, nectarml.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        _d_real = D_real_loss.mean().item()
        _d_fake = D_fake_loss.mean().item()
        
        ### DISCRIMINATOR (BACKWARD) ###
        
        disc.zero_grad()
        
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        ### GENERATOR (FORWARD) ###
        
        D_fake = disc(x, y_fake)
        G_fake_loss = BCE(D_fake, nectarml.ones_like(D_fake))
        L1 = L1_LOSS(y_fake, y) * L1_LAMBDA
        G_loss = G_fake_loss + L1
        
        _g_gan = G_fake_loss.mean().item()
        _G_l1 = L1.mean().item()
        
        ### GENERATOR (BACKWARD) ###
        
        opt_gen.zero_grad()
        
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()            
        
        ### POST-ITER ###
        
        if idx % 10 == 0: 
            print(f'Iteration: {idx}')
            print(f'Loss:')
            print(f'    D_real: {_d_real}')
            print(f'    D_fake: {_d_fake}')
            print(f'    G_GAN:  {_g_gan}')
            print(f'    G_L1:   {_G_l1}')

def main():
    disc = Discriminator(in_channels=3).to(DEVICE)
    gen = Generator(in_channels=3).to(DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    
    ### LOAD CHECKPOINTS ###
    
    train_dataset = Pix2pixDataset(TRAIN_SET_PATH)
    train_loader = Dataloader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    g_scaler = nectarml.amp.GradScaler()
    d_scaler = nectarml.amp.GradScaler()
    
    val_dataset = Pix2pixDataset(VAL_SET_PATH)
    val_loader = Dataloader(val_dataset, batch_size=1, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch+1}')
        train_fn(
            disc, gen, 
            train_loader, 
            opt_disc, opt_gen, 
            L1_LOSS, BCE, 
            g_scaler, d_scaler
        )
        
        if epoch % EXAMPLE_SAVE_RATE == 0:
            ### SAVE EXAMPLES ###
            pass
        if epoch % MODEL_SAVE_RATE == 0:
            ### SAVE CHECKPOINT ###
            pass
        
if __name__ == '__main__':
    main()


