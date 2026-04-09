from itertools import product

import numpy as np
from PIL import Image

import nectarml
from nectarml.vision import transforms as T

SHAPE = (1, 3, 16, 16)

def _run_transform_test(transforms: list[T.Transform]) -> None:
    for device, transform in product(['cpu', 'cuda'], transforms):
        x = nectarml.rand(SHAPE).to(device)
        transform(x)

def test_spatial() -> None:
    _run_transform_test([
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
    ])
    
def test_color() -> None:
    _run_transform_test([
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
    ])

def test_blur() -> None:
    _run_transform_test([
        T.GaussianBlur(p=1.0), 
        T.MotionBlur(p=1.0),
        T.MedianBlur(p=1.0),
        T.BoxBlur(p=1.0),
        T.RandomBlur(p=1.0),
        T.Sharpen(p=1.0), 
        T.Emboss(p=1.0),
        T.UnsharpMask(p=1.0)
    ])

def test_noise() -> None:
    _run_transform_test([
        T.GaussianNoise(p=1.0),
        T.SaltAndPepperNoise(p=1.0),
        T.SpeckleNoise(p=1.0),
        T.ISONoise(p=1.0), 
        T.MultiplicativeNoise(p=1.0),
        T.ImageCompression(p=1.0)
    ])
    
def test_erasing() -> None:
    _run_transform_test([
        T.Erasing(p=1.0),
        T.CoarseDropout(p=1.0),
        T.GridDropout(p=1.0),
        T.RandomLensFlare(p=1.0),
        T.RandomFog(p=1.0), 
        T.RandomRain(p=1.0),
        T.RandomSnow(p=1.0),
        T.RandomShadow(p=1.0),
        T.Spatter(p=1.0)
    ])

def test_filter() -> None:
    _run_transform_test([
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
    ])

def test_normalization() -> None:
    _run_transform_test([
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
        T.Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
        T.MinMaxNormalize(), 
        T.ToFloat(), 
        T.ToUint8()
    ])
    
def test_format() -> None:
    rng = np.random.default_rng()
    arr = rng.random(SHAPE)
    img = Image.fromarray(arr.squeeze(0).transpose((1, 2, 0)).astype(np.uint8))
    
    x = T.ToTensor()(arr)
    T.ToTensor()(img)
    
    T.ToPIL()(x)
    T.ToNumpy()(x) 
    
    y = T.ToTorch()(x)
    T.FromTorch()(y) 
     
    T.ConvertDtype()(x)
    z = T.ChangeDevice('cuda')(x)
    T.ChangeDevice('cpu')(z)
    
    T.ToCPU(z) 
    T.ToCUDA(x)
    
    T.Cast('cuda', new_dtype=nectarml.int32)(x)
    T.Cast('cpu', new_dtype=nectarml.int32)(z)
    
    T.ToContiguous()(x)
    T.ToContiguous()(z)
    
def test_composition() -> None:
    transforms = [
        T.ChannelShuffle(p=1),
        T.GaussianBlur(p=1),
        T.ToSepia(p=1)
    ]
    _run_transform_test([
        T.Compose(transforms),
        T.RandomApply(transforms, p=0),
        T.RandomApply(transforms, p=1),
        T.RandomChoice(transforms, p=0),
        T.RandomChoice(transforms, p=1),
        T.RandomOrder(transforms),
        T.OneOf(transforms)
    ])

def test_utility() -> None:
    _run_transform_test([
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
        T.MaskedFill(mask=nectarml.ones(SHAPE)), 
        T.Morphological(operation='dilation', p=1.0),
        T.Morphological(operation='erosion', p=1.0),
        T.OverlayElements(element=nectarml.rand((1, 3, 4, 4)), p=1.0), 
        T.OverlayText(text='a', font_size=12, p=1.0)
    ])
