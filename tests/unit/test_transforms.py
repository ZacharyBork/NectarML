from itertools import product

import pytest
import numpy as np
from PIL import Image

import nectarml
from nectarml.vision import transforms as T

### TRANSFORMS ###

SPATIAL = [
    T.Pad(padding=(1, 1)), 
    T.RandomCrop((2, 2), p=1.0), 
    T.CenterCrop((2, 2)),
    T.RandomResizedCrop((2, 2), p=1.0),
    T.Resize((8, 8)), 
    T.RandomHorizontalFlip(p=1.0),
    T.RandomVerticalFlip(p=1.0),
    T.Transpose(p=1.0),
    T.Rotate(p=1.0), 
    T.RandomRotation(p=1.0),
    T.RandomRotate90(p=1.0),
    T.RandomAffine(p=1.0),
    T.RandomPerspective(p=1.0), 
    T.ElasticTransform(p=1.0), 
    T.GridDistortion(p=1.0),
    T.OpticalDistortion(p=1.0),
    T.Swirl(p=1.0)
]

COLOR = [
    T.ColorJitter(p=1.0), 
    T.RandomBrightness(p=1.0),
    T.RandomContrast(p=1.0),
    T.RandomSaturation(p=1.0),
    T.RandomHue(p=1.0),
    T.RandomGamma(p=1.0),
    T.ToGrayscale(p=1.0),
    T.ToBlackAndWhite(p=1.0),
    T.ToSepia(p=1.0),
    T.Equalize(p=1.0),
    T.AutoContrast(p=1.0), 
    T.Solarize(p=1.0), 
    T.Posterize(p=1.0),
    T.Invert(p=1.0),
    T.CLAHE(p=1.0),
    T.ChannelShuffle(p=1.0),
    T.ChannelDropout(p=1.0), 
    T.RGBShift(p=1.0), 
    T.HueSaturationValue(p=1.0),
    T.TonemapHDR(p=1.0),
    T.ChromaticAberration(p=1.0), 
    T.Vignetting(p=1.0), 
    T.Illumination(p=1.0),
    T.Quantize(p=1.0)
]

BLUR = [
    T.GaussianBlur(p=1.0), 
    T.MotionBlur(p=1.0),
    T.MedianBlur(p=1.0),
    T.BoxBlur(p=1.0),
    T.RandomBlur(p=1.0),
    T.Sharpen(p=1.0), 
    T.Emboss(p=1.0),
    T.UnsharpMask(p=1.0)
]

NOISE = [
    T.GaussianNoise(p=1.0),
    T.SaltAndPepperNoise(p=1.0),
    T.SpeckleNoise(p=1.0),
    T.ISONoise(p=1.0), 
    T.MultiplicativeNoise(p=1.0),
    T.ImageCompression(p=1.0)
]

ERASING = [
    T.Erasing(p=1.0),
    T.CoarseDropout(p=1.0),
    T.GridDropout(p=1.0),
    T.RandomLensFlare(p=1.0),
    T.RandomFog(p=1.0), 
    T.RandomRain(p=1.0),
    T.RandomSnow(p=1.0),
    T.RandomShadow(p=1.0),
    T.Spatter(p=1.0)
]

FILTER = [
    T.Convolve(p=1.0), 
    T.Sobel(p=1.0),
    T.Prewitt(p=1.0),
    T.Laplacian(p=1.0),
    T.Dither(p=1.0),
    T.Halftone(p=1.0),
    T.Kuwahara(p=1.0),
    T.Pixelate(p=1.0),
    T.AsciiRender(p=1.0),
    T.DifferenceOfGaussians(p=1.0)
]

NORMALIZATION = [
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
    T.Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
    T.MinMaxNormalize(), 
    T.ToFloat(), 
    T.ToUint8()
]

UTILITY = [
    T.NoOp(), 
    T.Resample(scale_factor=(2.0, 2.0)), 
    T.Derivative(mode='ddx'),
    T.Derivative(mode='ddy'),
    T.Derivative(mode='ddx', per_channel=True),
    T.Derivative(mode='ddy', per_channel=True),
    T.UVMap(),
    T.NormalMap(),
    T.Permute((0, 1, 3, 2)),
    T.Clamp(min_value=0.0, max_value=1.0),
    T.MaskedFill(mask=nectarml.ones((1, 3, 16, 16))), 
    T.Morphological(operation='dilation', p=1.0),
    T.Morphological(operation='erosion', p=1.0),
    T.OverlayElements(element=nectarml.rand((1, 3, 4, 4)), p=1.0), 
    T.OverlayText(text='a', font_size=12, p=1.0)
]


@pytest.mark.parametrize('transforms', [
    SPATIAL, COLOR, BLUR, NOISE, ERASING, FILTER, NORMALIZATION, UTILITY])
def test_transforms(
    transforms: list[T.Transform], 
    sample_input2d: np.ndarray
) -> None:
    for device, transform in product(['cpu', 'cuda'], transforms):
        x = nectarml.Tensor(sample_input2d).to(device)
        transform(x)

def test_composition(sample_input2d: np.ndarray) -> None:
    transforms = [T.ChannelShuffle(p=1), T.GaussianBlur(p=1), T.ToSepia(p=1)]
    COMPOSITION = [
        T.Compose(transforms),
        T.RandomApply(transforms, p=0),
        T.RandomApply(transforms, p=1),
        T.RandomChoice(transforms, p=0),
        T.RandomChoice(transforms, p=1),
        T.RandomOrder(transforms),
        T.OneOf(transforms)
    ]
    for device, transform in product(['cpu', 'cuda'], COMPOSITION):
        x = nectarml.Tensor(sample_input2d).to(device)
        transform(x)

def test_format(sample_input2d: np.ndarray) -> None:
    arr = sample_input2d
    img = Image.fromarray(arr.squeeze(0).transpose((1, 2, 0)).astype(np.uint8))
    
    nectar_tensor_cpu = T.ToTensor()(arr)
    T.ToTensor()(img)
    
    nectar_tensor_cuda = T.ChangeDevice('cuda')(nectar_tensor_cpu)
    T.ChangeDevice('cpu')(nectar_tensor_cuda)
    
    T.ToPIL()(nectar_tensor_cpu)
    T.ToPIL()(nectar_tensor_cuda)
    T.ToNumpy()(nectar_tensor_cpu) 
    T.ToNumpy()(nectar_tensor_cuda)
    
    T.ConvertDtype()(nectar_tensor_cpu)
    T.ConvertDtype()(nectar_tensor_cuda)
    
    T.Cast('cuda', new_dtype=nectarml.int32)(nectar_tensor_cpu)
    T.Cast('cpu', new_dtype=nectarml.int32)(nectar_tensor_cuda)
    
    T.ToContiguous()(nectar_tensor_cpu)
    T.ToContiguous()(nectar_tensor_cuda)
    
    T.ToCPU(nectar_tensor_cuda) 
    T.ToCUDA(nectar_tensor_cpu)
    
    torch_tensor = T.ToTorch()(nectar_tensor_cpu)
    T.FromTorch()(torch_tensor) 
     
