import nectarml
import nectarml.nn as nn

class Block(nn.Module):
    def __init__(
        self,
        in_channels:   int,
        out_channels:  int,
        id_downsample: nn.Module | None = None,
        stride:        int = 1
    ) -> None:
        super().__init__()
        self.expansion     = 4
        self.id_downsample = id_downsample
        self.relu = nn.ReLU()
        
        layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            self.relu
        )
        layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            self.relu
        )
        layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, 1, 1, 0),
            nn.BatchNorm2d(out_channels*self.expansion)
        )
        self.layers = nn.ModuleList([layer1, layer2, layer3])

    def forward(self, x: nectarml.Tensor) -> nectarml.Tensor:
        identity = x
        for layer in self.layers: x = layer(x)

        if self.id_downsample is not None:
            identity = self.id_downsample(identity)
            
        return self.relu(x + identity)

class ResNet(nn.Module):
    def __init__(
        self,
        layers:         list[int],
        image_channels: int,
        num_classes:    int
    ) -> None:
        super().__init__()
        self.in_ch   = 64
        self._input = nn.Sequential(
            nn.Conv2d(image_channels, 64, 3, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(3, 2, 1)
        )
        
        self.layers = nn.ModuleList([
            self._layer(layers[0], out_channels=64,  stride=1), 
            self._layer(layers[1], out_channels=128, stride=2), 
            self._layer(layers[2], out_channels=256, stride=2), 
            self._layer(layers[3], out_channels=512, stride=2)
        ])
        self.head = nn.Sequential(
            nn.Lambda(func=lambda x : x.mean(dim=(2, 3))),
            nn.Linear(512*4, num_classes)
        )
        
    def _layer(
        self, 
        num_residuals: int,
        out_channels:  int,
        stride:        int
    ) -> nn.Sequential:
        id_downsample = None
        layers        = []
             
        if stride != 1 or self.in_ch != out_channels * 4:
            id_downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_channels * 4, 1, stride),
                nn.BatchNorm2d(out_channels * 4))     
        
        layers.append(Block(self.in_ch, out_channels, id_downsample, stride))
        
        self.in_ch = out_channels * 4
        for _ in range(num_residuals - 1):
            layers.append(Block(self.in_ch, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: nectarml.Tensor) -> nectarml.Tensor:
        x = self._input(x)
        for layer in self.layers: x = layer(x)
        return self.head(x)

def ResNet50(image_channels: int = 3, num_classes: int = 1000):
    return ResNet([3, 4, 6, 3], image_channels, num_classes)

def ResNet101(image_channels: int = 3, num_classes: int = 1000):
    return ResNet([3, 4, 23, 3], image_channels, num_classes)

def ResNet152(image_channels: int = 3, num_classes: int = 1000):
    return ResNet([3, 8, 36, 3], image_channels, num_classes)
