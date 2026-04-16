import math
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
from pyfastnoiselite.pyfastnoiselite import \
    FastNoiseLite, NoiseType, FractalType

import nectarml.functional as F
from nectarml          import typing
from nectarml.tensor   import Tensor
from nectarml.creation import zeros, rand, ones, linspace
from nectarml.vision.transforms.transform import Transform 
from nectarml.vision.transforms.common    import TransformInput, lerp

class Erasing(Transform):
    def __init__(
        self,
        scale: tuple[float, float] = (0.02, 0.33),
        ratio: tuple[float, float] = (0.3, 3.3),
        fill:                float = 0.0,
        erase_mask:           bool = False,
        p:                   float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.scale = scale
        self.ratio = ratio
        self.fill = fill
        self.erase_mask = erase_mask
    
    def _build_mask(self, input: Tensor) -> Tensor:
        B, C, H, W = input.shape
        mask = ones((B, 1, H, W), input.dtype, input.device)
        image_area = H * W
        
        for b in range(B):
            area = self._scale[b] * image_area
            aspect_ratio = max(self._epsilon, self._ratio[b])

            hole_h = min(int(math.sqrt(area / aspect_ratio)), H)
            hole_w = min(int(math.sqrt(area * aspect_ratio)), W)
            
            cy = int(self._cy[b] * H)
            cx = int(self._cx[b] * W)
            
            pY = (max(0, cy - hole_h // 2), min(H, cy + hole_h // 2))
            pX = (max(0, cx - hole_w // 2), min(W, cx + hole_w // 2))
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mask[b, 0, pY[0]:pY[1], pX[0]:pX[1]] = 0.0
        
        return mask
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        mask = self._build_mask(input)
        return input * mask + self.fill * input.max().item() * (1 - mask)
    
    def _build_parameters(self, batches: int) -> None:
        self._scale = []
        self._ratio = []
        self._cy = []
        self._cx = []
        for _ in range(batches):
            self._scale.append(self._random_in_range(self.scale))
            self._ratio.append(self._random_in_range(self.ratio))
            self._cy.append(self._random_in_range((0.0, 1.0)))
            self._cx.append(self._random_in_range((0.0, 1.0)))
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters(input.image.shape[0])
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) 
                        if self.erase_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        ) 

class CoarseDropout(Transform):
    def __init__(
        self,
        num_holes_range:    tuple[int,   int]   = (1, 2),
        holes_height_range: tuple[float, float] = (0.1, 0.2),
        holes_width_range:  tuple[float, float] = (0.1, 0.2),
        fill:               float = 0.0,
        erase_mask:          bool = False,
        p:                  float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.num_holes_range = num_holes_range
        self.holes_height_range = holes_height_range
        self.holes_width_range = holes_width_range
        self.fill = fill
        self.erase_mask = erase_mask
    
    def _build_mask(self, input: Tensor) -> Tensor:
        B, C, H, W = input.shape
        mask = ones((B, 1, H, W), input.dtype, input.device)
        
        for b in range(B):
            num_holes = self._num_holes[b]
            for hole in range(num_holes):
                hole_h = int(self._hole_h[b][hole]*H)
                hole_w = int(self._hole_w[b][hole]*W)
                
                cy = int(self._cy[b][hole] * H)
                cx = int(self._cx[b][hole] * W)
                
                pY = (max(0, cy - hole_h // 2), min(H, cy + hole_h // 2))
                pX = (max(0, cx - hole_w // 2), min(W, cx + hole_w // 2))
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mask[b, 0, pY[0]:pY[1], pX[0]:pX[1]] = 0.0
        
        return mask
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        mask = self._build_mask(input)
        return input * mask + self.fill * input.max().item() * (1 - mask)
     
    def _build_parameters(self, batches: int) -> None:
        self._num_holes = []
        self._hole_h = []
        self._hole_w = []
        self._cy = []
        self._cx = []
        for _ in range(batches):
            num_holes = int(round(self._random_in_range(self.num_holes_range)))
            self._num_holes.append(num_holes)
            self._hole_h.append(
                [self._random_in_range(self.holes_height_range) 
                 for _ in range(num_holes)])
            self._hole_w.append(
                [self._random_in_range(self.holes_width_range)
                 for _ in range(num_holes)])
            self._cy.append(
                [self._random_in_range((0.0, 1.0)) 
                 for _ in range(num_holes)])
            self._cx.append(
                [self._random_in_range((0.0, 1.0)) 
                 for _ in range(num_holes)])
            
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters(input.image.shape[0])
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) 
                        if self.erase_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        ) 

class GridDropout(Transform):
    def __init__(
        self,
        ratio:           float = 0.5,
        random_offset:   bool  = True,
        holes_number_xy: tuple[int,   int]   = (10, 10),
        shift_xy:        tuple[float, float] = (0.0, 0.0),
        fill:            float = 0.0,
        erase_mask:      bool  = False,
        p:               float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.ratio = ratio
        self.random_offset = random_offset
        for num in holes_number_xy:
            assert num > 1, 'Hole counts must be greater than 1 on both axes.'
        self.holes_number_xy = holes_number_xy
        self.shift_xy = shift_xy
        self.fill = fill
        self.erase_mask = erase_mask
    
    def _build_mask(self, input: Tensor) -> Tensor:
        B, C, H, W = input.shape
        mask = ones((B, 1, H, W), input.dtype, input.device)

        for b in range(B):
            for x in range(self.holes_number_xy[0]):
                for y in range(self.holes_number_xy[1]):
                    grid_y = H / self.holes_number_xy[1]
                    grid_x = W / self.holes_number_xy[0]
                    
                    hole_h = int(grid_y * self.ratio)
                    hole_w = int(grid_x * self.ratio)
                    
                    cy = int(y / (self.holes_number_xy[1]-1) * H)
                    cx = int(x / (self.holes_number_xy[0]-1) * W)
                    
                    if not self.random_offset:
                        cy += self.shift_xy[1] * H
                        cx += self.shift_xy[0] * W
                    else:
                        offset_y = self._offsets[b][x][y][0] * (grid_y // 2)
                        offset_x = self._offsets[b][x][y][1] * (grid_x // 2)
                        cy += int((offset_y - (offset_y * 0.5)) * 2)
                        cx += int((offset_x - (offset_x * 0.5)) * 2)
                    
                    pY = (max(0, cy - hole_h // 2), min(H, cy + hole_h // 2))
                    pX = (max(0, cx - hole_w // 2), min(W, cx + hole_w // 2))
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mask[b, 0, pY[0]:pY[1], pX[0]:pX[1]] = 0.0
        
        return mask
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        mask = self._build_mask(input)
        return input * mask + self.fill * input.max().item() * (1 - mask)
    
    def _build_parameters(self, batches: int) -> None:
        self._offsets = []
        for _ in range(batches):
            offset = {}
            for x in range(self.holes_number_xy[0]):
                offset[x] = {}
                for y in range(self.holes_number_xy[1]):
                    offset[x][y] = (
                        self._random_in_range((0.0, 1.0)),
                        self._random_in_range((0.0, 1.0)))
            self._offsets.append(offset)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters(input.image.shape[0])
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = self._transform(input.mask) 
                        if self.erase_mask else input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        ) 

class RandomLensFlare(Transform):
    def __init__(
        self,
        num_ghosts:         int   = 4,
        ghost_radius_range: tuple[float, float] = (0.01, 0.06),
        ghost_alpha_range:  tuple[float, float] = (0.1,  0.4),
        halo_radius:        float = 0.15,
        halo_alpha:         float = 0.1,
        streak_count:       int   = 6,
        streak_alpha:       float = 0.15,
        streak_length:      float = 0.3,
        chromatic_shift:    float = 2.0,
        glow_radius:        float = 0.05,
        global_scale:       float = 1.0,
        source_position:    tuple[float, float] | None = None,
        p:                  float = 0.5
    ) -> None:
        '''
        Based on the lens flare algorithm described in this paper:
            - https://resources.mpi-inf.mpg.de/lensflareRendering/pdf/flare.pdf
        '''
        super().__init__(p=p)
        self.num_ghosts = num_ghosts
        self.ghost_radius_range = tuple(
            [i*global_scale for i in ghost_radius_range])
        self.ghost_alpha_range = ghost_alpha_range
        self.halo_radius = halo_radius * global_scale
        self.halo_alpha = halo_alpha
        self.streak_count = streak_count
        self.streak_alpha = streak_alpha
        self.streak_length = streak_length * global_scale
        self.chromatic_shift = chromatic_shift
        self.glow_radius = glow_radius * global_scale
        self.source_position = source_position

    def _make_ghost(
        self, 
        cx: float, cy: float, 
        radius: float, 
        alpha: float,
        chroma_shift: float
    ) -> Tensor:
        ghost = zeros(self.layer_shape, dtype=self._dtype, device=self._device)

        for c, shift in enumerate([-chroma_shift, 0, chroma_shift]):
            cx_c = cx + shift
            dist = ((self.proj_x - cx_c)**2 + (self.proj_y - cy)**2).sqrt()
            r_px = radius * self.smaller_side
            mask = (1.0 - dist / r_px).clamp(0.0, 1.0) ** 2
            ghost[c] = mask * alpha
        
        return ghost

    def _make_halo(self, cx: float, cy: float) -> Tensor:
        halo = zeros(self.layer_shape, dtype=self._dtype, device=self._device)
        
        dist = ((self.proj_x - cx)**2 + (self.proj_y - cy)**2).sqrt()
        r_px = self.halo_radius * self.smaller_side
        ring_width = max(self._epsilon, (2 * (r_px * 0.2)**2))
        ring = (-((dist - r_px) ** 2) / ring_width).exp()
        halo[:] = ring.unsqueeze(0).expand(self.layer_shape) * self.halo_alpha
        return halo

    def _make_streaks(self, cx: float, cy: float) -> Tensor:
        streaks = zeros(
            self.layer_shape, dtype=self._dtype, device=self._device)

        for i in range(self.streak_count):
            angle = (i / self.streak_count) * 3.1415926535
            dx = math.cos(angle)
            dy = math.sin(angle)
            
            proj = (self.proj_x - cx) * dx + (self.proj_y - cy) * dy
            perp = (self.proj_x - cx) * (-dy) + (self.proj_y - cy) * dx
            
            length_px = max(
                self._epsilon, self.streak_length * self.smaller_side)
            streak_width = 1.5
            
            length_mask = (1.0 - proj.abs() / length_px).clamp(0.0, 1.0) ** 2
            width_mask = (-(perp ** 2) / (2 * streak_width ** 2)).exp()
            
            streak = length_mask * width_mask * self.streak_alpha
            streaks[:] += streak.unsqueeze(0).expand(self.layer_shape)
        
        return streaks.clamp(0.0, 1.0)

    def _make_glow(self, cx: float, cy: float, alpha: float) -> Tensor:
        glow = zeros(self.layer_shape, dtype=self._dtype, device=self._device)
                
        dist = ((self.proj_x - cx)**2 + (self.proj_y - cy)**2).sqrt()
        r_px = max(
            self._epsilon, self.glow_radius * self.smaller_side)
        soft = (-(dist ** 2) / (2 * r_px ** 2)).exp()
        glow[:] = soft.unsqueeze(0).expand(self.layer_shape) * alpha
        return glow

    def _build_projections(self, H: int, W: int) -> None:
        self.proj_y = linspace(0, H-1, H, self._dtype, self._device)
        self.proj_y = self.proj_y.reshape((H, 1)).expand((H, W))
        self.proj_x = linspace(0, W-1, W, self._dtype, self._device)
        self.proj_x = self.proj_x.reshape((1, W)).expand((H, W))

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        self._device = input.device
        self._dtype = input.dtype
        
        B, C, H, W = input.shape
        max_value = input.max().item()
        
        self._build_projections(H, W)
        self.layer_shape = (3, H, W)
        self.smaller_side = min(H, W)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for b in range(B):
                flare = zeros(
                    (3, H, W), dtype=self._dtype, device=self._device)
                cx, cy = W/2, H/2
                axis_dx, axis_dy = cx-self._sx, cy-self._sy

                flare += self._make_glow(self._sx, self._sy, alpha=0.8) \
                       + self._make_halo(self._sx, self._sy) \
                       + self._make_streaks(self._sx, self._sy)

                for _ in range(self.num_ghosts):
                    gx = (self._sx + axis_dx * self._t)
                    gy = (self._sy + axis_dy * self._t)
                    
                    flare += self._make_ghost(
                        gx, gy, self._radius, self._alpha, self._shift)

                flare = flare.clamp(0.0, 1.0) * max_value
                input[b] = (input[b] + flare).clamp(0.0, max_value)

        return input

    def _build_parameters(self, H: int, W: int) -> None:
        if self.source_position is not None:
            self._sx = self.source_position[0] * W
            self._sy = self.source_position[1] * H
        else:
            self._sx = self.rng.uniform(0.1, 0.9) * W
            self._sy = self.rng.uniform(0.0, 0.4) * H
        self._t = self.rng.uniform(0.3, 2.0)
        
        self._radius = self._random_in_range(self.ghost_radius_range)
        self._alpha  = self._random_in_range(self.ghost_alpha_range)
        self._shift  = self.rng.uniform(0, self.chromatic_shift)

    def forward(self, input: TransformInput) -> TransformInput:
        _, _, H, W = input.image.shape
        self._build_parameters(H, W)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
        
class RandomFog(Transform):
    def __init__(
        self,
        scale:           int = 100,
        octaves:         int = 4,
        intensity_range: tuple[float, float]  = (0.3, 0.7),
        fog_color:       tuple[int, int, int] = (255, 255, 255),
        p:               float = 0.5
    ) -> None:
        super().__init__(p=p)
        assert scale > 0, 'RandomFog "scale" must be > 0.'
        self.scale = scale
        self.octaves = octaves
        self.intensity_range = intensity_range
        self.fog_color = fog_color
        
    def _perlin_approx(
        self, 
        H: int, 
        W: int,
        device: typing.DeviceLikeType,
        dtype: typing.dtype
    ) -> Tensor:
        noise = zeros((1, 1, H, W), dtype, device)
        amplitude = frequency = 1.0
        
        for _ in range(self.octaves):
            octave_h = max(1, int(H * frequency / self.scale))
            octave_w = max(1, int(W * frequency / self.scale))
            octave = F.upsample(
                rand((1, 1, octave_h, octave_w), self._seed, dtype, device),
                size=(H, W), mode='bilinear')
            
            noise = noise + amplitude * octave
            amplitude *= 0.5
            frequency *= 2.0
        
        min_value = noise.min().item()
        divisor = max(self._epsilon, noise.max().item() - min_value)
        return (noise - min_value) / divisor

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        B, C, H, W = input.shape
        max_value = input.max().item()
        
        fog_maps = F.stack(
            [self._perlin_approx(H, W, input.device, input.dtype).squeeze(0)
            for _ in range(B)], dim=0)
        
        fog_color = Tensor(self.fog_color, (3,), input.dtype, input.device)
        fog_color = fog_color.reshape((1, 3, 1, 1))

        fog = (1 - fog_maps * self._intensity) \
            + fog_color * fog_maps * self._intensity
        
        out = input + fog
        divisor = max(self._epsilon, out.max().item() * max_value)
        return (out / divisor).clamp(0.0, max_value)

    def _build_parameters(self) -> None:
        self._intensity = self._random_in_range(self.intensity_range)
        self._seed = int(self._random_in_range((0.0, 999999999.0)))

    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
        
class RandomRain(Transform):
    def __init__(
        self,
        brightness_coef: float | tuple[float, float] = (0.5, 0.8),
        num_drops:       int   | tuple[int,   int]   = (100, 300),
        drop_length:     int   | tuple[int,   int]   = (20,  35),
        drop_width:      int   | tuple[int,   int]   = (1,   2),
        drop_color:      tuple[int, int, int] = (200, 200, 200),
        blur_value:      int   = 3,
        p:               float = 0.5
    ) -> None:
        '''
        Reference:
            - https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
        '''
        super().__init__(p=p)
        self.brightness_coef = (brightness_coef, brightness_coef) \
            if isinstance(brightness_coef, int | float) else brightness_coef
        self.num_drops = (num_drops, num_drops) \
            if isinstance(num_drops, int) else num_drops
        self.drop_length = (drop_length, drop_length) \
            if isinstance(drop_length, int) else drop_length
        self.drop_width = (drop_width, drop_width) \
            if isinstance(drop_width, int) else drop_width
        self.drop_color = drop_color
        self.blur_value = blur_value
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        _, C, H, W = input.shape
        out = input.cpu() * self._brightness_coef
        color = self._color.to(out.device, input.dtype)
        color = (
            color / 255 
          * max(self._epsilon, input.max().item()) 
          * self._brightness_coef)
        
        for i in range(self._num_drops):
            x, y = self._xs[i-1], self._ys[i-1]
            angle = self._angles[i-1]
            
            dy = self._lengths[i-1]
            dx = int(math.sin(math.radians(angle)) * dy)
            
            x2 = max(0.0, min(x + dx, W - 1))
            y2 = max(0.0, min(y + dy, H - 1))
            
            steps = max(abs(x2 - x), abs(y2 - y))
            if steps == 0: continue
            
            length = math.sqrt((x2-x)**2 + (y2-y)**2) + 1e-8
            perp_x = -(y2 - y) / length
            perp_y =  (x2 - x) / length
            half_w = self._drop_width // 2

            for s in range(int(steps)):
                t = s / steps
                cx = int(x + t * (x2 - x))
                cy = int(y + t * (y2 - y))
                
                for w in range(-half_w, half_w + 1):
                    px = int(cx + w * perp_x)
                    py = int(cy + w * perp_y)
                    if 0 <= px < W and 0 <= py < H:
                        for c in range(C):
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                out[:, c, py, px] = color[:, c, py, px] \
                                    if c < 3 else out[:, c, py, px]
            
        return out.to(input.device)
    
    def _build_parameters(self, H: int, W: int) -> None:
        self._brightness_coef = self._random_in_range(self.brightness_coef)
        self._num_drops = int(round(self._random_in_range(self.num_drops)))
        self._drop_length = int(round(self._random_in_range(self.drop_length)))
        self._drop_width = int(round(self._random_in_range(self.drop_width)))
        
        self._color = Tensor(self.drop_color, dtype=typing.float32)
        self._color = self._color.view((1, 3, 1, 1)).expand((1, 3, H, W))
        
        self._xs = []
        self._ys = []
        self._angles = []
        self._lengths = []
        for _ in range(self._num_drops):
            self._xs.append(self.rng.integers(0, W))
            self._ys.append(self.rng.integers(0, H))
            self._angles.append(self.rng.uniform(-15, 15))
            self._lengths.append(
                int(round(self._random_in_range(self.drop_length))))
    
    def forward(self, input: TransformInput) -> TransformInput:
        _, _, H, W = input.image.shape
        self._build_parameters(H, W)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class RandomSnow(Transform):
    def __init__(
        self,
        brighness_coef:   float = 1.5,
        snow_point_range: tuple[float, float] = (0.1, 0.3),
        p:                float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.brightness_coef = brighness_coef
        self.snow_point_range = snow_point_range
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item() + 1e-8
        norm = input / max_value
        
        r, g, b = norm.unbind(dim=1)
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b).unsqueeze(1)
        
        snow_mask = ((luminance-self._snow_point) / (1.0-self._snow_point))
        snow_mask = snow_mask.clamp(0.0, 1.0)
        
        noise = rand(snow_mask.shape, self._seed, input.dtype, input.device)
        snow_mask = (snow_mask + noise * 0.1).clamp(0.0, 1.0)
        
        out = norm + snow_mask * (1.0 - norm) * self.brightness_coef
        return (out * max_value).clamp(0.0, max_value)
    
    def _build_parameters(self) -> None:
        self._snow_point = self._random_in_range(self.snow_point_range)
        self._seed = int(self._random_in_range((0.0, 999999999.0)))
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._build_parameters()
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class RandomShadow(Transform):
    def __init__(
        self,
        shadow_intensity:  float | tuple[float, float] = (0.3,  0.7),
        noise_frequency:   float | tuple[float, float] = (0.01, 0.04),
        noise_threshold:   float | tuple[float, float] = (0.1,  0.4),
        blur_sigma:        float | tuple[float, float] = (15.0, 40.0),
        color_shift:       float | tuple[float, float] = (0.0,  0.05),
        falloff_intensity: float | tuple[float, float] = (1.0,  1.5),
        falloff_contrast:  float | tuple[float, float] = (0.8,  1.6),
        p:                 float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.shadow_intensity = (shadow_intensity, shadow_intensity) \
            if isinstance(shadow_intensity, int|float) else shadow_intensity
        self.noise_frequency = (noise_frequency, noise_frequency) \
            if isinstance(noise_frequency, int|float) else noise_frequency
        self.noise_threshold = (noise_threshold, noise_threshold) \
            if isinstance(noise_threshold, int|float) else noise_threshold
        self.blur_sigma = (blur_sigma, blur_sigma) \
            if isinstance(blur_sigma, int|float) else blur_sigma
        self.color_shift = (color_shift, color_shift) \
            if isinstance(color_shift, int|float) else color_shift
        self.falloff_intensity = (falloff_intensity, falloff_intensity) \
            if isinstance(falloff_intensity, int|float) else falloff_intensity
        self.falloff_contrast = (falloff_contrast, falloff_contrast) \
            if isinstance(falloff_contrast, int|float) else falloff_contrast
    
    def _make_cloud_mask(self, H: int, W: int) -> None:
        fn = FastNoiseLite(seed=int(self.rng.integers(0, 2**31)))
        fn.noise_type         = NoiseType.NoiseType_OpenSimplex2
        fn.frequency          = self._frequency
        fn.fractal_type       = FractalType.FractalType_FBm
        fn.fractal_octaves    = 4
        fn.fractal_gain       = 0.5
        fn.fractal_lacunarity = 2.0

        ramp = np.linspace(0.0, 1.0, H, dtype=np.float32)
        ramp = np.broadcast_to(ramp[:, np.newaxis], (H, W))
        ramp = ramp * self._falloff_intensity

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        noise = np.vectorize(fn.get_noise)(xx, yy).astype(np.float32)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        
        mask = (noise > self._threshold).astype(np.float32)
        mask = gaussian_filter(mask, sigma=self._sigma)
        if mask.max() > 0: mask = mask / (mask.max() + 1e-8)
        mask = mask * ramp**self._falloff_contrast
        
        self._mask = Tensor(mask, mask.shape, dtype=typing.float32)
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input

        out    = input.clone()
        mask   = self._mask.unsqueeze(0).to(input.device, input.dtype)
        shadow = 1.0 - self._intensity * mask
        out    = out * shadow
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if self._shift > 0.0 and input.shape[1] >= 3:
                amt = self._intensity * mask
                out[0] = out[0] * (1.0-amt[0] * self._shift)
                out[1] = out[1] * (1.0-amt[0] * self._shift * 0.5)
                out[2] = (out[2] + amt[0] * self._shift * 0.1).clamp(0.0, 1.0)

        return out
    
    def _build_parameters(self, H: int, W: int) -> None:
        self._intensity = self._random_in_range(self.shadow_intensity)
        self._frequency = self._random_in_range(self.noise_frequency)
        self._threshold = self._random_in_range(self.noise_threshold)
        self._sigma = self._random_in_range(self.blur_sigma)
        self._shift = self._random_in_range(self.color_shift)
        self._falloff_intensity = self._random_in_range(self.falloff_intensity)
        self._falloff_contrast = self._random_in_range(self.falloff_contrast)
        
        self._make_cloud_mask(H, W)
    
    def forward(self, input: TransformInput) -> TransformInput:
        _, _, H, W = input.image.shape
        self._build_parameters(H, W)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class Spatter(Transform):
    def __init__(
        self,
        droplet_scales:     int | tuple[int, int] = (3, 5),
        droplet_density:    float | tuple[float, float] = (0.08, 0.12),
        droplet_color:      tuple[int, int, int] = (255, 255, 255),
        droplet_refraction: float = 0.15,
        p:                  float = 0.5
    ) -> None:
        '''
        Adapted from Élie Michel's awesome GLSL shader, found here:
        - https://www.shadertoy.com/view/ldSBWW
        '''
        super().__init__(p=p)
        self.droplet_scales = (droplet_scales, droplet_scales) \
            if isinstance(droplet_scales, int) else droplet_scales
        self.droplet_density = (droplet_density, droplet_density) \
            if isinstance(droplet_density, float | int) else droplet_density
        self.droplet_color = droplet_color
        self.droplet_refraction = droplet_refraction

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        _, C, H, W = input.shape
        out = input.clone()
        
        u = linspace(0, 1, W).unsqueeze(0)
        v = linspace(0, 1, H).unsqueeze(1)
        u = u.broadcast_to((H, W)).to(input.device)
        v = v.broadcast_to((H, W)).to(input.device)
        
        n_x = F.upsample(
            self._base_noise[0].to(input.device), 
            size=(H, W), mode='bilinear')
        n_x = n_x.squeeze(0).squeeze(0)
        n_y = F.upsample(
            self._base_noise[1].to(input.device), 
            size=(H, W), mode='bilinear')
        n_y = n_y.squeeze(0).squeeze(0)   
        
        drop_mask = zeros((H, W), dtype=input.dtype).to(input.device)
        disp_x    = zeros((H, W), dtype=input.dtype).to(input.device)
        disp_y    = zeros((H, W), dtype=input.dtype).to(input.device)
        
        for r in range(self._droplet_scales, 0, -1):
            ds_cur = self._ds[r-1]
            d = Tensor(ds_cur, dtype=typing.float32, device=input.device)
            ds = F.unbind(d, dim=0)
            
            ux, uy = u * self._gx, v * self._gy
            cell_x = (ux-0.25).round().to(dtype=typing.int32)
            cell_x = cell_x.clamp(0, self._n_cx-1)
            cell_y = (uy-0.25).round().to(dtype=typing.int32)
            cell_y = cell_y.clamp(0, self._n_cy-1)
            d_r, d_g = ds[0][cell_y, cell_x], ds[1][cell_y, cell_x]
                        
            phase = d_g
            p_x = 6.28 * ux + (n_x - 0.5) * 2.0
            p_y = 6.28 * uy + (n_y - 0.5) * 2.0
            s_x, s_y = (p_x + phase * 6.28).sin(), (p_y + phase * 6.28).sin()
            
            t = (s_x + s_y) * F.maximum(1.0 - d_g * 2.0, 0.0)
            
            active = (d_r < (5 - r) * self._droplet_density)
            active = active.to(dtype=typing.float32)
            active = F.minimum(active, (t > 0.5).to(dtype=typing.float32))
            
            cos_px = (p_x + phase * 6.28).cos()
            cos_py = (p_y + phase * 6.28).cos()
            
            nx, ny = -cos_px, -cos_py
            nz = F.where(active, (2.0 * (t - 0.5)).clamp(0.2, 2.0), 1.0)
            norm = (nx**2 + ny**2 + nz**2).sqrt() + 1e-8
            nx /= norm
            ny /= norm
            
            disp_x    = F.where(active, nx, disp_x)
            disp_y    = F.where(active, ny, disp_y)
            drop_mask = F.where(active, 1.0, drop_mask)

        src_x = (u - disp_x * self.droplet_refraction).clamp(0, 1)
        src_y = (v - disp_y * self.droplet_refraction).clamp(0, 1)
        
        px = (src_x * (W - 1)).clamp(0, W-1).to(dtype=typing.int32)
        py = (src_y * (H - 1)).clamp(0, H-1).to(dtype=typing.int32)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for c in range(C):
                refracted = input[:, c].squeeze(0)[py, px]
                out[:, c] = F.where(drop_mask > 0, refracted, out[:, c])
        
        color = (self._color/255).to(input.device, input.dtype)
        out = lerp(out, out * color, drop_mask)
        return out.to(input.device, input.dtype)

    def _build_parameters(self, H: int, W: int) -> None:
        self._droplet_scales = int(
            round(self._random_in_range(self.droplet_scales)))
        self._droplet_density = self._random_in_range(self.droplet_density)
        
        noise_scale = max(1, int(min(H, W) * 0.1))
        self._base_noise = [
            rand((1, 1, noise_scale, noise_scale)).to(dtype=typing.float32),
            rand((1, 1, noise_scale, noise_scale)).to(dtype=typing.float32)]
        
        self._color = Tensor(self.droplet_color, dtype=typing.float32)
        self._color = self._color.view((1, 3, 1, 1)).expand((1, 3, H, W))    
    
        self._ds = []
        for r in range(self._droplet_scales, 0, -1):
            self._gy, self._gx = H * r * 0.015, W * r * 0.015
            self._n_cy = max(1, int(self._gy))
            self._n_cx = max(1, int(self._gx))
            self._ds.append(self.rng.random(
                (2, self._n_cy, self._n_cx)).astype(np.float32))

    def forward(self, input: TransformInput) -> TransformInput:
        _, _, H, W = input.image.shape
        self._build_parameters(H, W)
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
