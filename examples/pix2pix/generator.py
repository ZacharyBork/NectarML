import nectarml
import nectarml.nn as nn

class Block(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        down: bool = True,
        activation: str = 'relu',
        use_dropout: bool = False
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                4, 2, 1, bias=False, padding_mode='reflect')
            if down
            else nn.ConvTranspose2d(
                in_channels, out_channels,
                4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        )
        
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: nectarml.Tensor) -> nectarml.Tensor:
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    
class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        features: int = 64
    ) -> None:
        super().__init__()
        
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        
        self.down1 = Block(features,   features*2, activation='leaky')
        self.down2 = Block(features*2, features*4, activation='leaky')
        self.down3 = Block(features*4, features*8, activation='leaky')
        self.down4 = Block(features*8, features*8, activation='leaky')
        self.down5 = Block(features*8, features*8, activation='leaky')
        self.down6 = Block(features*8, features*8, activation='leaky')
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode='reflect'),
            nn.ReLU()
        )
        
        self.up1 = Block(features*8,   features*8, down=False, use_dropout=True)
        self.up2 = Block(features*8*2, features*8, down=False, use_dropout=True)
        self.up3 = Block(features*8*2, features*8, down=False, use_dropout=True)
        self.up4 = Block(features*8*2, features*8, down=False)
        self.up5 = Block(features*8*2, features*4, down=False)
        self.up6 = Block(features*4*2, features*2, down=False)
        self.up7 = Block(features*2*2, features,   down=False)
    
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x: nectarml.Tensor) -> nectarml.Tensor:
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
    
        bottleneck = self.bottleneck(d7)
        
        up1 = self.up1(bottleneck)
        up2 = self.up2(nectarml.cat([up1, d7], dim=1))
        up3 = self.up3(nectarml.cat([up2, d6], dim=1))
        up4 = self.up4(nectarml.cat([up3, d5], dim=1))
        up5 = self.up5(nectarml.cat([up4, d4], dim=1))
        up6 = self.up6(nectarml.cat([up5, d3], dim=1))
        up7 = self.up7(nectarml.cat([up6, d2], dim=1))
        return self.final_up(nectarml.cat([up7, d1], dim=1))
    

if __name__ == '__main__':
    x = nectarml.randn((1, 3, 256, 256)).cuda()
    model = Generator(in_channels=3, features=64).cuda()
    y = model(x)
    print(y.shape)
        
