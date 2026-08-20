import nectarml
from   nectarml import nn

class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()
 
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc_mu      = nn.LazyLinear(latent_dim)
        self.fc_logvar  = nn.LazyLinear(latent_dim)
 
    def forward(self, x: nectarml.Tensor) -> nectarml.Tensor:
        x = self.encode(x).flatten(start_dim=1)
        return self.fc_mu(x), self.fc_logvar(x).clamp(-4.0, 15.0)

class Decoder(nn.Module):
    def __init__(
        self, 
        out_channels: int,
        latent_dim:   int, 
        latent_size:  int
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.linear      = nn.Linear(latent_dim, 256 * self.latent_size**2)
        self.decode      = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
 
    def forward(self, z: nectarml.Tensor) -> nectarml.Tensor:
        x = self.linear(z).view(-1, 256, self.latent_size, self.latent_size)
        return self.decode(x)

class VAE(nn.Module):
    def __init__(
        self, 
        in_channels:  int = 3, 
        spatial_size: int = 64,
        latent_dim:   int = 128,
    ) -> None:
        '''Variational autoencoder model.

        Args:
            in_channels  : The number of channels in the input tensors.
            spatial_size : The spatial size (HW) of the input tensors.
            latent_dim   : The number of features at the bottleneck between the
                           encoder and decoder.
        '''
        super().__init__() # First we super().__init__() the base Module class.
        
        # Then we initialize the encoder layer.
        self.encoder = Encoder(in_channels, latent_dim)

        # And finally, the decoder layer. We calculate the latent spatial size
        # as `input_size // 2**num_conv_layers`. Since we are using a fixed 
        # number of convolutional layers (4), we can simplify this to 
        # `input_size // 2**4`, or `input_size // 16`.
        self.decoder = Decoder(in_channels, latent_dim, spatial_size // 16)

    def _init_weights(self, module: nn.Module) -> None:
        '''Initializes model weights.

        Sets all bias parameters to zeros and initializes the weights of the
        the projection heads with smaller values than default.
        
        Args:
            module : The module to initialize the weights for.
        '''
        if module is self.encoder.fc_mu or module is self.encoder.fc_logvar:
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if module.bias is not None: nn.init.zeros_(module.bias)

    def reparameterize(
        self, 
        mu:     nectarml.Tensor, 
        logvar: nectarml.Tensor
    ) -> nectarml.Tensor:
        std = nectarml.exp(0.5 * logvar)
        eps = nectarml.randn_like(std, requires_grad=False)
        return mu + std * eps
 
    def forward(
        self, 
        x: nectarml.Tensor
    ) -> tuple[nectarml.Tensor, nectarml.Tensor, nectarml.Tensor]:
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decoder(z)
        return x_hat, mu, logvar
 
    @nectarml.no_grad()
    def sample(self, n: int) -> nectarml.Tensor:
        device   = self.parameters()[0].device
        dtype    = self.parameters()[0].dtype
        features = self.decoder.linear.in_features
        z        = nectarml.randn(n, features, dtype=dtype, device=device)
        return self.decoder(z)




