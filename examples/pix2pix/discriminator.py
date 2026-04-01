import nectarml
import nectarml.nn as nn

class CNNBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 2
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=4, stride=stride, 
                bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x: nectarml.Tensor) -> nectarml.Tensor:
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3,
        features: list[int] = [64, 128, 256, 512]
    ) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels*2, features[0],
                kernel_size=4, stride=2, padding=1, 
                padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(
                    in_channels, feature, 
                    stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
            
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, 
                padding=1, padding_mode='reflect')
        )
        
        self.model = nn.Sequential(*layers)
    
    def forward(
        self, 
        x: nectarml.Tensor, 
        y: nectarml.Tensor
    ) -> nectarml.Tensor:
        x = nectarml.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)
    
    
    
if __name__ == '__main__':
    x = nectarml.rand((1, 3, 256, 256)).cuda()
    y = nectarml.rand((1, 3, 256, 256)).cuda()

    model = Discriminator().to('cuda')
    z = model(x, y)
    print(z.shape)

