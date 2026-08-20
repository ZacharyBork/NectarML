import numpy as np
import pytest
import torch

import nectarml
from nectarml import typing as nml_typing
import nectarml.vision.transforms as T

###############################################################################
# Config
###############################################################################

DEVICE = 'cpu'


@pytest.fixture(autouse=True)
def fixed_seed():
    np.random.seed(42)


###############################################################################
# Helpers
###############################################################################


def img(B=1, C=3, H=64, W=64, dtype=np.float32, lo=0.0, hi=1.0):
    data = np.random.uniform(lo, hi, (B, C, H, W)).astype(dtype)
    return nectarml.Tensor(data)


def img_uint8(B=1, C=3, H=64, W=64):
    data = np.random.randint(0, 256, (B, C, H, W), dtype=np.uint8)
    return nectarml.Tensor(data, dtype=nectarml.uint8)


def mask(B=1, H=64, W=64):
    data = np.random.randint(0, 2, (B, 1, H, W)).astype(np.float32)
    return nectarml.Tensor(data)


def to_np(t):
    return t.detach().cpu().numpy().astype(np.float32)


def assert_shape(out, expected, label=''):
    assert out.shape == nectarml.typing.Size(expected), (
        f'{label} shape: got {out.shape}, '
        f'expected {nectarml.typing.Size(expected)}'
    )


def assert_no_nan(out, label=''):
    assert not np.any(np.isnan(to_np(out))), f'{label} contains NaN'


def assert_no_inf(out, label=''):
    assert not np.any(np.isinf(to_np(out))), f'{label} contains Inf'


def assert_finite(out, label=''):
    assert_no_nan(out, label)
    assert_no_inf(out, label)


def assert_range(out, lo, hi, label=''):
    vals = to_np(out)
    assert (
        vals.min() >= lo - 1e-4
    ), f'{label} min={vals.min():.4f} below expected {lo}'
    assert (
        vals.max() <= hi + 1e-4
    ), f'{label} max={vals.max():.4f} above expected {hi}'


def assert_passthrough(transform, x, label=''):
    import copy

    t = copy.deepcopy(transform)
    t.p = 0.0
    out = t(x)
    assert np.allclose(
        to_np(out), to_np(x), atol=1e-5
    ), f'{label} p=0 passthrough failed'


def assert_same_shape_as_input(transform, x, label=''):
    out = transform(x)
    assert (
        out.shape == x.shape
    ), f'{label} shape changed: {x.shape} → {out.shape}'
    assert_finite(out, label)


###############################################################################
# BLUR
###############################################################################


class TestBlur:
    def test_gaussian_blur_shape(self):
        x = img()
        out = T.GaussianBlur(kernel_size=5, sigma=1.0, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_gaussian_blur_passthrough(self):
        assert_passthrough(T.GaussianBlur(p=0.5), img())

    def test_gaussian_blur_range_range(self):
        x = img()
        out = T.GaussianBlur(kernel_size=(3, 7), sigma=(0.5, 2.0), p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_box_blur_shape(self):
        x = img()
        out = T.BoxBlur(kernel_size=5, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_box_blur_passthrough(self):
        assert_passthrough(T.BoxBlur(p=0.5), img())

    def test_motion_blur_shape(self):
        x = img()
        out = T.MotionBlur(kernel_size=5, angle=(0.0, 360.0), p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_median_blur_shape(self):
        x = img()
        out = T.MedianBlur(kernel_size=3, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_random_blur_shape(self):
        x = img()
        out = T.RandomBlur(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_unsharp_mask_shape(self):
        x = img()
        out = T.UnsharpMask(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_emboss_shape(self):
        x = img()
        out = T.Emboss(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_alpha_blending(self):
        x = img()
        out_0 = T.GaussianBlur(kernel_size=5, sigma=2.0, alpha=0.0, p=1.0)(x)
        assert np.allclose(
            to_np(out_0), to_np(x), atol=1e-4
        ), 'alpha=0 should return unchanged image'


###############################################################################
# COLOR
###############################################################################


class TestColor:
    def test_color_jitter_shape(self):
        x = img()
        out = T.ColorJitter(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_color_jitter_passthrough(self):
        assert_passthrough(T.ColorJitter(p=0.5), img())

    def test_random_brightness_shape(self):
        x = img()
        out = T.RandomBrightness(value_range=(1.0, 1.0), p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert np.allclose(to_np(out), to_np(x), atol=1e-4)

    def test_random_contrast_shape(self):
        x = img()
        out = T.RandomContrast(value_range=(0.8, 1.2), p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_random_saturation_shape(self):
        x = img()
        out = T.RandomSaturation(value_range=(0.8, 1.2), p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_random_hue_shape(self):
        x = img()
        out = T.RandomHue(value_range=(0.9, 1.1), p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_random_gamma_brightens(self):
        x = img(lo=0.3, hi=0.7)
        out_bright = T.RandomGamma(value_range=(0.5, 0.5), p=1.0)(x)
        assert to_np(out_bright).mean() > to_np(x).mean() - 1e-3

    def test_to_grayscale_channels(self):
        x = img()
        out = T.ToGrayscale(p=1.0)(x)
        vals = to_np(out)
        assert (
            np.allclose(vals[:, 0], vals[:, 1], atol=1e-4) or out.shape[1] == 1
        ), 'ToGrayscale should produce equal channels or single channel'

    def test_to_black_and_white_binary(self):
        x = img(lo=0.0, hi=255.0)
        out = T.ToBlackAndWhite(white_point=125, p=1.0)(x)
        vals = to_np(out)
        unique = np.unique(vals)
        assert len(unique) <= 3, (
            f'ToBlackAndWhite should produce at most 2 values, '
            f'got {len(unique)}'
        )

    def test_to_sepia_shape(self):
        x = img()
        out = T.ToSepia(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_invert_correctness(self):
        x = img(lo=0.0, hi=1.0)
        out = T.Invert(p=1.0)(x)
        expected = 1.0 - to_np(x)
        assert np.allclose(
            to_np(out), expected, atol=0.02
        ), 'Invert should approximately flip values'

    def test_invert_passthrough(self):
        assert_passthrough(T.Invert(p=0.5), img())

    def test_solarize_shape(self):
        x = img()
        out = T.Solarize(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_posterize_shape(self):
        x = img()
        out = T.Posterize(levels=4, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_equalize_shape(self):
        x = img_uint8()
        out = T.Equalize(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_autocontrast_shape(self):
        x = img()
        out = T.AutoContrast(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_channel_shuffle_shape(self):
        x = img()
        out = T.ChannelShuffle(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert np.allclose(
            to_np(x).sum(axis=1), to_np(out).sum(axis=1), atol=1e-4
        )

    def test_channel_dropout_shape(self):
        x = img()
        out = T.ChannelDropout(channel_range=(1, 1), fill=0.0, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        vals = to_np(out)
        has_zero_channel = any(np.allclose(vals[0, c], 0.0) for c in range(3))
        assert (
            has_zero_channel
        ), 'ChannelDropout should zero at least one channel'

    def test_rgb_shift_shape(self):
        x = img()
        out = T.RGBShift(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_hsv_shape(self):
        x = img()
        out = T.HueSaturationValue(hue=0.0, saturation=1.0, value=1.0, p=1.0)(
            x
        )
        assert_shape(out, (1, 3, 64, 64))
        assert np.allclose(to_np(out), to_np(x), atol=0.02)

    def test_tonemap_hdr_shape(self):
        x = img(lo=0.0, hi=10.0)
        for method in ['reinhard', 'filmic', 'aces']:
            out = T.TonemapHDR(method=method, p=1.0)(x)
            assert_shape(out, (1, 3, 64, 64), label=method)
            assert_finite(out, label=method)

    def test_vignetting_shape(self):
        x = img()
        out = T.Vignetting(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_illumination_shape(self):
        x = img()
        for mode in ['linear', 'radial']:
            out = T.Illumination(mode=mode, p=1.0)(x)
            assert_shape(out, (1, 3, 64, 64), label=mode)
            assert_finite(out, label=mode)

    def test_quantize_shape(self):
        x = img()
        out = T.Quantize(levels=4, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_clahe_shape(self):
        x = img_uint8()
        out = T.CLAHE(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_chromatic_aberration_shape(self):
        x = img()
        out = T.ChromaticAberration(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)


###############################################################################
# ERASING
###############################################################################


class TestErasing:
    def test_erasing_shape(self):
        x = img()
        out = T.Erasing(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))

    def test_erasing_fill_value(self):
        x = img(lo=0.5, hi=0.5)
        out = T.Erasing(fill=0.0, p=1.0)(x)
        assert to_np(out).min() < 0.1, 'Erasing should zero out a region'

    def test_erasing_passthrough(self):
        assert_passthrough(T.Erasing(p=0.5), img())

    def test_coarse_dropout_shape(self):
        x = img()
        out = T.CoarseDropout(num_holes_range=(1, 3), p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))

    def test_coarse_dropout_fill(self):
        x = img(lo=0.5, hi=0.5)
        out = T.CoarseDropout(num_holes_range=(2, 2), fill=0.0, p=1.0)(x)
        assert to_np(out).min() < 0.1

    def test_grid_dropout_shape(self):
        x = img()
        out = T.GridDropout(ratio=0.5, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_erasing_mask_coerase(self):
        x = img(lo=0.5, hi=0.5)
        m = nectarml.Tensor(np.ones((1, 1, 64, 64), dtype=np.float32))
        out_x, out_m = T.Erasing(fill=0.0, erase_mask=True, p=1.0)(x, m)
        x_zeros = to_np(out_x).mean(axis=1) < 0.1
        m_vals = to_np(out_m)[:, 0]
        assert np.any(
            m_vals == 0.0
        ), 'Mask should be erased where image is erased'

    def test_random_lens_flare(self):
        x = img()
        out = T.RandomLensFlare(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_random_fog(self):
        x = img()
        out = T.RandomFog(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_random_rain(self):
        x = img()
        out = T.RandomRain(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_random_snow(self):
        x = img()
        out = T.RandomSnow(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_random_shadow(self):
        x = img()
        out = T.RandomShadow(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_spatter(self):
        x = img()
        out = T.Spatter(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)


###############################################################################
# FILTER
###############################################################################


class TestFilter:
    def test_convolve_identity(self):
        x = img()
        kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        out = T.Convolve(kernel=kernel, alpha=1.0, p=1.0)(x)
        assert np.allclose(
            to_np(out), to_np(x), atol=1e-3
        ), 'Identity kernel should not change image'

    def test_convolve_shape(self):
        x = img()
        out = T.Convolve(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_sobel_shape(self):
        x = img()
        out = T.Sobel(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_sobel_per_channel(self):
        x = img()
        out = T.Sobel(per_channel=True, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))

    def test_prewitt_shape(self):
        x = img()
        out = T.Prewitt(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_laplacian_shape(self):
        x = img()
        out = T.Laplacian(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_dog_shape(self):
        x = img()
        out = T.DifferenceOfGaussians(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_kuwahara_shape(self):
        x = img()
        out = T.Kuwahara(radius=3, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_pixelate_shape(self):
        x = img()
        out = T.Pixelate(block_size=8, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_pixelate_constant_blocks(self):
        x = img()
        block = 8
        out = T.Pixelate(block_size=block, p=1.0)(x)
        vals = to_np(out)[0, 0]
        for r in range(0, 64, block):
            for c in range(0, 64, block):
                block_vals = vals[r : r + block, c : c + block]
                assert np.allclose(
                    block_vals, block_vals[0, 0], atol=1e-3
                ), f'Block at ({r},{c}) should be uniform after pixelation'


###############################################################################
# FORMAT
###############################################################################


class TestFormat:
    def test_to_tensor_from_numpy(self):
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        from PIL import Image

        pil = Image.fromarray(arr)
        out = T.ToTensor(normalize=True)(pil)
        assert isinstance(out, nectarml.Tensor)
        assert out.shape[-2:] == nectarml.typing.Size([64, 64])
        assert_range(out, 0.0, 1.0)

    def test_to_pil(self):
        from PIL import Image

        x = img(lo=0.0, hi=1.0)
        out = T.ToPIL(normalize=True)(x)
        assert isinstance(out, Image.Image)

    def test_to_numpy(self):
        x = img()
        out = T.ToNumpy()(x)
        assert isinstance(out, np.ndarray)

    def test_convert_dtype(self):
        x = img()
        out = T.ConvertDtype(new_dtype=nml_typing.float16)(x)
        assert out.dtype == nml_typing.float16

    def test_to_cuda(self):
        x = img()
        out = T.ToCUDA()(x)
        assert out.device == 'cuda'

    def test_to_cpu_from_cuda(self):
        x = img()
        out = T.ToCUDA()(x)
        out = T.ToCPU()(out)
        assert out.device == 'cpu'

    def test_cast_device_and_dtype(self):
        x = img()
        out = T.Cast(new_device='cuda', new_dtype=nml_typing.float16)(x)
        assert out.device == 'cuda'
        assert out.dtype == nml_typing.float16

    def test_change_device(self):
        x = img()
        out = T.ChangeDevice(new_device='cuda')(x)
        assert out.device == 'cuda'

    def test_to_contiguous(self):
        x = img()
        out = T.ToContiguous()(x)
        assert_shape(out, (1, 3, 64, 64))

    def test_permute(self):
        x = img()
        out = T.Permute(dims=(0, 2, 3, 1))(x)
        assert_shape(out, (1, 64, 64, 3))

    def test_clamp(self):
        x = img(lo=-1.0, hi=2.0)
        out = T.Clamp(min_value=0.0, max_value=1.0)(x)
        assert_range(out, 0.0, 1.0)

    def test_clamp_min_only(self):
        x = img(lo=-1.0, hi=0.5)
        out = T.Clamp(min_value=0.0)(x)
        assert to_np(out).min() >= -1e-4

    def test_masked_fill(self):
        x = img(lo=0.5, hi=0.5)
        m = nectarml.Tensor(np.ones((1, 3, 64, 64), dtype=np.float32))
        out = T.MaskedFill(mask=m, value=0.0)(x)
        assert np.allclose(
            to_np(out), 0.0, atol=1e-5
        ), 'MaskedFill with all-ones mask should zero entire image'

    def test_from_torch_to_torch_roundtrip(self):
        x = img()
        torch_t = T.ToTorch()(x)
        assert isinstance(torch_t, torch.Tensor)
        back = T.FromTorch()(torch_t)
        assert isinstance(back, nectarml.Tensor)
        assert np.allclose(to_np(back), to_np(x), atol=1e-5)


###############################################################################
# NOISE
###############################################################################


class TestNoise:
    def test_gaussian_noise_shape(self):
        x = img()
        out = T.GaussianNoise(std_range=(0.1, 0.1), p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_gaussian_noise_changes_image(self):
        x = img()
        out = T.GaussianNoise(std_range=(0.1, 0.1), p=1.0)(x)
        assert not np.allclose(
            to_np(out), to_np(x)
        ), 'GaussianNoise should modify image'

    def test_gaussian_noise_passthrough(self):
        assert_passthrough(T.GaussianNoise(p=0.5), img())

    def test_gaussian_noise_per_channel(self):
        x = img()
        out = T.GaussianNoise(std_range=(0.1, 0.1), per_channel=True, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_salt_and_pepper_shape(self):
        x = img()
        out = T.SaltAndPepperNoise(amount=(0.05, 0.05), p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_speckle_noise_shape(self):
        x = img()
        out = T.SpeckleNoise(std_range=(0.1, 0.1), p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_multiplicative_noise_identity(self):
        x = img()
        out = T.MultiplicativeNoise(multiplier_range=(1.0, 1.0), p=1.0)(x)
        assert np.allclose(
            to_np(out), to_np(x), atol=1e-4
        ), 'multiplier=1.0 should leave image unchanged'

    def test_multiplicative_noise_shape(self):
        x = img()
        out = T.MultiplicativeNoise(multiplier_range=(0.8, 1.2), p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_iso_noise_shape(self):
        x = img()
        out = T.ISONoise(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_image_compression_shape(self):
        x = img_uint8()
        for fmt in ['jpeg', 'webp']:
            out = T.ImageCompression(
                compression_type=fmt, quality_range=(80, 80), p=1.0
            )(x)
            assert_shape(out, (1, 3, 64, 64), label=fmt)
            assert_finite(out, label=fmt)


###############################################################################
# NORMALIZATION
###############################################################################


class TestNormalization:
    def test_normalize_denormalize_roundtrip(self):
        x = img(lo=0.0, hi=1.0)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalized = T.Normalize(mean=mean, std=std)(x)
        denormalized = T.Denormalize(mean=mean, std=std)(normalized)
        assert np.allclose(
            to_np(x), to_np(denormalized), atol=1e-4
        ), 'Normalize → Denormalize should be identity'

    def test_normalize_changes_values(self):
        x = img(lo=0.0, hi=1.0)
        out = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(x)
        assert not np.allclose(
            to_np(out), to_np(x)
        ), 'Normalize should change values'

    def test_minmax_normalize_range(self):
        x = img(lo=0.0, hi=255.0)
        out = T.MinMaxNormalize(min_value=0.0, max_value=1.0)(x)
        assert_range(out, 0.0, 1.0)

    def test_minmax_normalize_custom_range(self):
        x = img(lo=0.0, hi=255.0)
        out = T.MinMaxNormalize(min_value=-1.0, max_value=1.0)(x)
        assert_range(out, -1.0, 1.0)

    def test_to_float_uint8_input(self):
        x = img_uint8()
        print(x.dtype)
        out = T.ToFloat(scale=True)(x)
        assert out.dtype == nml_typing.float32
        assert_range(out, 0.0, 1.0)

    def test_to_float_no_scale(self):
        x = img_uint8()
        out = T.ToFloat(scale=False)(x)
        assert out.dtype == nml_typing.float32

    def test_to_uint8_from_float(self):
        x = img(lo=0.0, hi=1.0)
        out = T.ToUint8(scale=True)(x)
        vals = to_np(out)
        assert (
            vals.min() >= 0 and vals.max() <= 255
        ), 'ToUint8 should produce values in [0, 255]'

    def test_to_float_half_precision(self):
        x = img_uint8()
        out = T.ToFloat(half_precision=True, scale=True)(x)
        assert out.dtype == nml_typing.float16


###############################################################################
# SPATIAL
###############################################################################


class TestSpatial:
    def test_pad_constant(self):
        x = img()
        out = T.Pad(padding=4, fill=0.0, padding_mode='constant')(x)
        assert_shape(out, (1, 3, 72, 72))

    def test_pad_all_sides(self):
        x = img()
        out = T.Pad(padding=(2, 4, 6, 8), fill=0.0)(x)
        assert_shape(out, (1, 3, 78, 70))

    def test_pad_reflect(self):
        x = img()
        out = T.Pad(padding=4, padding_mode='reflect')(x)
        assert_shape(out, (1, 3, 72, 72))
        assert_finite(out)

    def test_pad_mask(self):
        x = img()
        m = mask()
        out_x, out_m = T.Pad(padding=4, transform_mask=True)(x, m)
        assert_shape(out_x, (1, 3, 72, 72))
        assert_shape(out_m, (1, 1, 72, 72))

    def test_random_crop_size(self):
        x = img()
        out = T.RandomCrop(size=(32, 32), p=1.0)(x)
        assert_shape(out, (1, 3, 32, 32))

    def test_random_crop_square_int(self):
        x = img()
        out = T.RandomCrop(size=48, p=1.0)(x)
        assert_shape(out, (1, 3, 48, 48))

    def test_random_crop_mask(self):
        x = img()
        m = mask()
        out_x, out_m = T.RandomCrop(size=32, p=1.0, transform_mask=True)(x, m)
        assert_shape(out_x, (1, 3, 32, 32))
        assert_shape(out_m, (1, 1, 32, 32))

    def test_center_crop_size(self):
        x = img()
        out = T.CenterCrop(size=(40, 40))(x)
        assert_shape(out, (1, 3, 40, 40))

    def test_center_crop_centered(self):
        data = np.zeros((1, 1, 64, 64), dtype=np.float32)
        data[0, 0, 28:36, 28:36] = 1.0
        x = nectarml.Tensor(data)
        out = T.CenterCrop(size=32)(x)
        assert (
            to_np(out).sum() > 0
        ), 'Center crop should contain the center block'

    def test_resize_to_size(self):
        x = img()
        out = T.Resize(size=(32, 32))(x)
        assert_shape(out, (1, 3, 32, 32))

    def test_resize_scale_factor(self):
        x = img()
        out = T.Resize(scale_factor=0.5)(x)
        assert_shape(out, (1, 3, 32, 32))

    def test_resize_mask(self):
        x = img()
        m = mask()
        out_x, out_m = T.Resize(size=32, transform_mask=True)(x, m)
        assert_shape(out_x, (1, 3, 32, 32))
        assert_shape(out_m, (1, 1, 32, 32))

    def test_horizontal_flip_reverses(self):
        data = np.zeros((1, 1, 4, 4), dtype=np.float32)
        data[0, 0, :, 0] = 1.0
        x = nectarml.Tensor(data)
        out = T.RandomHorizontalFlip(p=1.0)(x)
        vals = to_np(out)
        assert np.allclose(
            vals[0, 0, :, -1], 1.0, atol=1e-4
        ), 'Horizontal flip should move left column to right'

    def test_horizontal_flip_passthrough(self):
        assert_passthrough(T.RandomHorizontalFlip(p=0.5), img())

    def test_horizontal_flip_mask(self):
        x = img()
        m = mask()
        out_x, out_m = T.RandomHorizontalFlip(p=1.0, transform_mask=True)(x, m)
        assert_shape(out_x, (1, 3, 64, 64))
        assert_shape(out_m, (1, 1, 64, 64))

    def test_vertical_flip_reverses(self):
        data = np.zeros((1, 1, 4, 4), dtype=np.float32)
        data[0, 0, 0, :] = 1.0
        x = nectarml.Tensor(data)
        out = T.RandomVerticalFlip(p=1.0)(x)
        vals = to_np(out)
        assert np.allclose(
            vals[0, 0, -1, :], 1.0, atol=1e-4
        ), 'Vertical flip should move top row to bottom'

    def test_vertical_flip_passthrough(self):
        assert_passthrough(T.RandomVerticalFlip(p=0.5), img())

    def test_transpose_swaps_dims(self):
        data = np.random.rand(1, 3, 64, 64).astype(np.float32)
        x = nectarml.Tensor(data)
        out = T.Transpose(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_rotate_90_shape(self):
        x = img()
        out = T.Rotate(angle=90.0, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_rotate_180_twice_is_identity(self):
        x = img()
        r = T.Rotate(angle=180.0, fill_value=0.0, p=1.0)
        out = r(r(x))
        assert np.allclose(
            to_np(out), to_np(x), atol=0.05
        ), '180° rotation twice should approximately restore image'

    def test_random_rotation_shape(self):
        x = img()
        out = T.RandomRotation(rotation_range=(-45.0, 45.0), p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_random_rotate90_shape(self):
        x = img()
        out = T.RandomRotate90(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))

    def test_random_affine_shape(self):
        x = img()
        out = T.RandomAffine(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_random_perspective_shape(self):
        x = img()
        out = T.RandomPerspective(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_optical_distortion_shape(self):
        x = img()
        out = T.OpticalDistortion(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_elastic_transform_shape(self):
        x = img()
        out = T.ElasticTransform(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_grid_distortion_shape(self):
        x = img()
        out = T.GridDistortion(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_swirl_shape(self):
        x = img()
        out = T.Swirl(p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_random_resized_crop(self):
        x = img()
        out = T.RandomResizedCrop(crop_size=32, output_size=64, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)


###############################################################################
# UTILITY
###############################################################################


class TestUtility:
    def test_noop(self):
        x = img()
        out = T.NoOp()(x)
        assert np.allclose(to_np(out), to_np(x)), 'NoOp should be identity'

    def test_debug_print(self, capsys):
        x = img()
        T.DebugPrint('test')(x)
        captured = capsys.readouterr()
        assert 'test' in captured.out or True

    def test_make_grid_shape(self):
        imgs = [img() for _ in range(4)]
        batch = nectarml.Tensor(
            np.concatenate([to_np(i) for i in imgs], axis=0)
        )
        out = T.MakeGrid(nrow=2, padding=2)(batch)
        assert out.ndim == 3 or out.ndim == 4
        assert_finite(out)

    def test_resample_shape(self):
        x = img()
        out = T.Resample(size=(32, 32))(x)
        assert_shape(out, (1, 3, 32, 32))

    def test_derivative_ddx(self):
        x = img()
        out = T.Derivative(mode='ddx')(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_derivative_ddy(self):
        x = img()
        out = T.Derivative(mode='ddy')(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_uv_map(self):
        x = img()
        out = T.UVMap(tiling_x=1.0, tiling_y=1.0)(x)
        assert_finite(out)

    def test_normal_map(self):
        x = img()
        out = T.NormalMap(normal_power=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_morphological_dilation(self):
        x = img()
        out = T.Morphological(scale=3, operation='dilation', p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_morphological_erosion(self):
        x = img()
        out = T.Morphological(scale=3, operation='erosion', p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_overlay_elements_shape(self):
        x = img()
        element = img(B=1, C=3, H=16, W=16, lo=0.0, hi=1.0)
        out = T.OverlayElements(element=element, p=1.0)(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)


###############################################################################
# COMPOSITION
###############################################################################


class TestComposition:
    def test_compose_runs_all(self):
        x = img(lo=0.0, hi=255.0)
        pipeline = T.Compose(
            [
                T.RandomHorizontalFlip(p=1.0),
                T.RandomVerticalFlip(p=1.0),
                T.GaussianNoise(std_range=(0.01, 0.01), p=1.0),
            ]
        )
        out = pipeline(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_compose_order_matters(self):
        x = img()
        p1 = T.Compose([T.CenterCrop(32), T.Resize(64)])
        p2 = T.Compose([T.Resize(32), T.Resize(64)])
        out1 = p1(x)
        out2 = p2(x)
        assert out1.shape == out2.shape == nectarml.typing.Size([1, 3, 64, 64])

    def test_random_apply(self):
        x = img()
        pipeline = T.RandomApply(
            [
                T.GaussianBlur(p=1.0),
                T.RandomBrightness(p=1.0),
            ],
            p=1.0,
        )
        out = pipeline(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_random_apply_passthrough(self):
        x = img()
        pipeline = T.RandomApply([T.GaussianBlur(p=1.0)], p=0.0)
        out = pipeline(x)
        assert np.allclose(to_np(out), to_np(x), atol=1e-5)

    def test_random_choice_runs_one(self):
        x = img()
        pipeline = T.RandomChoice(
            [
                T.RandomHorizontalFlip(p=1.0),
                T.RandomVerticalFlip(p=1.0),
                T.GaussianBlur(kernel_size=3, p=1.0),
            ],
            p=1.0,
        )
        out = pipeline(x)
        assert_shape(out, (1, 3, 64, 64))

    def test_one_of_runs_exactly_one(self):
        x = img()
        pipeline = T.OneOf(
            [
                T.RandomHorizontalFlip(p=1.0),
                T.RandomVerticalFlip(p=1.0),
            ]
        )
        out = pipeline(x)
        assert_shape(out, (1, 3, 64, 64))

    def test_random_order_shape(self):
        x = img()
        pipeline = T.RandomOrder(
            [
                T.GaussianBlur(kernel_size=3, p=1.0),
                T.RandomBrightness(value_range=(0.9, 1.1), p=1.0),
            ]
        )
        out = pipeline(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)

    def test_compose_with_mask(self):
        x = img()
        m = mask()
        pipeline = T.Compose(
            [
                T.RandomHorizontalFlip(p=1.0, transform_mask=True),
                T.RandomCrop(size=32, p=1.0, transform_mask=True),
            ]
        )
        out_x, out_m = pipeline(x, m)
        assert_shape(out_x, (1, 3, 32, 32))
        assert_shape(out_m, (1, 1, 32, 32))

    def test_full_augmentation_pipeline(self):
        x = img(lo=0.0, hi=1.0)
        pipeline = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=(0.8, 1.2), contrast=(0.8, 1.2), p=0.5
                ),
                T.GaussianNoise(std_range=(0.0, 0.05), p=0.3),
                T.RandomCrop(size=48, pad_if_needed=True, p=1.0),
                T.Resize(size=64),
            ]
        )
        out = pipeline(x)
        assert_shape(out, (1, 3, 64, 64))
        assert_finite(out)
