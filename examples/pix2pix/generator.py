import nectarml
import nectarml.nn as nn

# First we define a block class, inheriting from nectarml.nn.Module.

class Block(nn.Module):
    def __init__(
        self, 
        in_channels:  int,
        out_channels: int,
        down:         bool = True,
        activation:   str  = 'relu',
        dropout:      bool = False
    ) -> None:
        super().__init__()
        
        # A Block consists of a single nectarml.nn.Sequential layer.
        # The if down=True, the layer uses a Conv2d, if False, it uses a
        # ConvTranspose2d. 
        
        # This is followed by instance normalization, then a non-linear 
        # activation function, the type of which is defined by "activation":
        # ("relu"=nn.ReLU(), "leaky"=nn.LeakyRelu(negative_slope=0.2))
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, 
                      padding=1, bias=False, padding_mode='reflect')
            if down
            else nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, 
                stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        )
        
        # Finally a dropout layer is defined.
        
        self.use_dropout = dropout
        self.dropout     = nn.Dropout(0.5)
        
    def forward(self, x: nectarml.Tensor) -> nectarml.Tensor:
        # Modules must define a forward() method!
        
        # Block.forward() takes input tensor x, runs it through the conv
        # Sequential, applies dropout if applicable, and returns the result.
        
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

# Then we define a Generator class with a UNet-style architecture.    
    
class Generator(nn.Module):
    
    # Generator.__init__() takes two arguments:
    #   1. in_channels : The number of input channels on the first conv layer.
    #   2. features    : The number of output channels on the first conv layer.
    
    def __init__(
        self,
        in_channels: int = 3,
        features:    int = 64
    ) -> None:
        super().__init__()
        
        # First we define the initial downsampling layer as:
        #   - Conv2d -> LeakyReLU(negative_slope=0.2)
        
        # This layer takes the feature count from "in_channels" to "features"
        # and downsamples the spatial resolution by a factor of 2.
        
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        
        # Then we use out Block class to define the subsequent downsampling 
        # layers, each of which doubles the feature count (up to a cap of
        # "features" * 8), and uses LeakyReLU for activation.
        
        self.down1 = Block(features,   features*2, activation='leaky')
        self.down2 = Block(features*2, features*4, activation='leaky')
        self.down3 = Block(features*4, features*8, activation='leaky')
        self.down4 = Block(features*8, features*8, activation='leaky')
        self.down5 = Block(features*8, features*8, activation='leaky')
        self.down6 = Block(features*8, features*8, activation='leaky')
        
        # Then we define a bottleneck layer as Conv2d -> ReLU
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode='reflect'),
            nn.ReLU()
        )
        
        # And then we use our Block class to define the decoder path. Feature
        # counts for each layer are reversed from the decoder path, and these
        # blocks use nn.ConvTranspose2d (via down=False) to upsample the 
        # spatial size of the tensor.
        
        # Note that input features here are multiplied by 2, to account for
        # the tensors from the skip connections.
        
        self.up1 = Block(features*8,   features*8, down=False, dropout=True)
        self.up2 = Block(features*8*2, features*8, down=False, dropout=True)
        self.up3 = Block(features*8*2, features*8, down=False, dropout=True)
        self.up4 = Block(features*8*2, features*8, down=False)
        self.up5 = Block(features*8*2, features*4, down=False)
        self.up6 = Block(features*4*2, features*2, down=False)
        self.up7 = Block(features*2*2, features,   down=False)
    
        # Lastly, we define a final upsampling layer, which return the tensor
        # to its original shape, and applies a Tanh activation, giving it an
        # output range of [-1:1].
    
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x: nectarml.Tensor) -> nectarml.Tensor:
        # Generator.forward() takes the input image tensor and first runs it
        # through the encode path layer by layer, feeding the output of one 
        # layer into the next.
        
        x  = self.initial_down(x)
        
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
    
        # We then run the resulf of the encoder path through the bottleneck
        # layer, and pass the output through to the decoder path.
    
        bottleneck = self.bottleneck(d6)
        
        # Next we work our way up through the decoder path. The first decoder
        # layer recieves the output of the bottleneck, then every subsequent
        # decoder layer recieves the output from the previous layer, 
        # concatenated with the output of the corresponding encoder layer.
        
        up1 = self.up1(bottleneck)
        up2 = self.up2(nectarml.cat([up1, d6], dim=1))
        up3 = self.up3(nectarml.cat([up2, d5], dim=1))
        up4 = self.up4(nectarml.cat([up3, d4], dim=1))
        up5 = self.up5(nectarml.cat([up4, d3], dim=1))
        up6 = self.up6(nectarml.cat([up5, d2], dim=1))
        up7 = self.up7(nectarml.cat([up6, d1], dim=1))
        
        # And lastly, we run the tensor through the final upsampling layer
        # and return the result.
        
        return self.final_up(nectarml.cat([up7, x], dim=1))

