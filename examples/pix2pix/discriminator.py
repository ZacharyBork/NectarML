import nectarml
import nectarml.nn as nn

# First we define a CNN block, inheriting from nectarml.nn.Module.

class CNNBlock(nn.Module):
    def __init__(
        self, 
        in_channels:  int,
        out_channels: int,
        stride:       int = 2
    ) -> None:
        super().__init__()
        
        # CNN block contains a single layer as a nectarml.nn.Sequential.
        # Layer is defined as: Conv2d -> BatchNorm -> LeakyRelu.
       
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=4, stride=stride, bias=False,
                padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
    def forward(self, x: nectarml.Tensor) -> nectarml.Tensor:
        # Modules must define a forward() method!
        
        # CNNBlock.forward() takes tensor x as input, runs it through the
        # conv Sequential, and returns the output.
        
        return self.conv(x)

# Next we define our discriminator class, also a nectarml.nn.Module.

class Discriminator(nn.Module):
    
    # Discriminator.__init__() takes two input arguments:
    # 
    #    1. in_channels : The number of input channels on the first conv layer.
    #    2. features    : The number of output features on each conv layer.
    
    def __init__(
        self, 
        in_channels: int = 3, 
        features:    list[int] = [64, 128, 256, 512]
    ) -> None:
        super().__init__()
        
        # First we define the initial layer, which increases feature count from
        # in_channels*2 (y, y_fake) to features[0], and downsamples spatial
        # size by a factor of 2.
        
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels*2, features[0],
                kernel_size=4, stride=2, padding=1, 
                padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        
        # Next we define the intermediate layers as a standard Python list.
        # Each subsequent layer continues downsampling and increasing feature
        # count until capping at 512 (features[-1]).
        
        layers = []
        in_channels = features[0]
        for out_channels in features[1:]:
            layers.append(
                CNNBlock(
                    in_channels, out_channels, 
                    stride=1 if out_channels == features[-1] else 2)
            )
            in_channels = out_channels
            
        # Then we will append a final conv module on the end which takes the
        # final feature count from our layers loop, and reduces it to 1, the
        # network's final prediction.
        
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, 
                padding=1, padding_mode='reflect')
        )
        
        # Finally we unpack our layers list into a nectarml.nn.Sequential.
        
        self.model = nn.Sequential(*layers)
   
    def forward(
        self, 
        x: nectarml.Tensor, 
        y: nectarml.Tensor
    ) -> nectarml.Tensor:
        
        # Discriminator.forward() takes two tensors, the real target and the
        # generator fake. It cononcatenates them along the feature dimension,
        # Then runs it through the layers defined above and returns the result.
        
        x = nectarml.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)
    
