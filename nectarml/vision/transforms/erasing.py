import math
import warnings
from typing import Literal

import nectarml.functional as F
from nectarml.tensor import Tensor
from nectarml.typing import DTypeLike, int32
from nectarml.creation import zeros, rand, ones, linspace
from nectarml.vision.transforms.transform import Transform, TransformInput 

class Erasing(Transform):
    def __init__(
        self,
        scale: tuple[float, float] = (0.02, 0.33),
        ratio: tuple[float, float] = (0.3, 3.3),
        fill: float = 0.0,
        erase_mask: bool = False
    ) -> None:
        super().__init__()
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
            aspect_ratio = self._ratio[b]

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
    
    def forward(self, input: TransformInput) -> TransformInput:
        B = input.image.shape[0]
        self._scale = []
        self._ratio = []
        self._cy = []
        self._cx = []
        for _ in range(B):
            self._scale.append(self._random_in_range(self.scale))
            self._ratio.append(self._random_in_range(self.ratio))
            self._cy.append(self._random_in_range((0.0, 1.0)))
            self._cx.append(self._random_in_range((0.0, 1.0)))
            
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
        num_holes_range: tuple[int, int] = (1, 2),
        holes_height_range: tuple[float, float] = (0.1, 0.2),
        holes_width_range: tuple[float, float] = (0.1, 0.2),
        fill: float = 0.0,
        erase_mask: bool = False
    ) -> None:
        super().__init__()
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
        
    def forward(self, input: TransformInput) -> TransformInput:
        B = input.image.shape[0]
        self._num_holes = []
        self._hole_h = []
        self._hole_w = []
        self._cy = []
        self._cx = []
        for _ in range(B):
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
        ratio: float = 0.5,
        random_offset: bool = True,
        holes_number_xy: tuple[int, int] = (10, 10),
        shift_xy: tuple[float, float] = (0.0, 0.0),
        fill: float = 0.0,
        erase_mask: bool = False
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.random_offset = random_offset
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
                        # offset_y = self._random_in_range((0, grid_y // 2))
                        # offset_x = self._random_in_range((0, grid_x // 2))
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
    
    def forward(self, input: TransformInput) -> TransformInput:
        B = input.image.shape[0]
        self._offsets = []
        for b in range(B):
            offset = {}
            for x in range(self.holes_number_xy[0]):
                offset[x] = {}
                for y in range(self.holes_number_xy[1]):
                    offset[x][y] = (
                        self._random_in_range((0.0, 1.0)),
                        self._random_in_range((0.0, 1.0)))
            self._offsets.append(offset)
                
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
        num_ghosts: int = 4,
        ghost_radius_range: tuple[float, float] = (0.01, 0.06),
        ghost_alpha_range: tuple[float, float] = (0.1, 0.4),
        halo_radius: float = 0.15,
        halo_alpha: float = 0.1,
        streak_count: int = 6,
        streak_alpha: float = 0.15,
        streak_length: float = 0.3,
        chromatic_shift: float = 2.0,
        glow_radius: float = 0.05,
        global_scale: float = 1.0,
        source_position: tuple[float, float] | None = None
    ) -> None:
        '''
        Based on the lens flare algorithm described in this paper:
            - https://resources.mpi-inf.mpg.de/lensflareRendering/pdf/flare.pdf
        '''
        super().__init__()
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
        ghost = zeros(self.layer_shape, dtype=self.dtype, device=self.device)

        for c, shift in enumerate([-chroma_shift, 0, chroma_shift]):
            cx_c = cx + shift
            dist = ((self.proj_x - cx_c)**2 + (self.proj_y - cy)**2).sqrt()
            r_px = radius * self.smaller_side
            mask = (1.0 - dist / r_px).clamp(0.0, 1.0) ** 2
            ghost[c] = mask * alpha
        
        return ghost

    def _make_halo(self, cx: float, cy: float) -> Tensor:
        halo = zeros(self.layer_shape, dtype=self.dtype, device=self.device)
        
        dist = ((self.proj_x - cx)**2 + (self.proj_y - cy)**2).sqrt()
        r_px = self.halo_radius * self.smaller_side
        ring_width = r_px * 0.2
        ring = (-((dist - r_px) ** 2) / (2 * ring_width ** 2)).exp()
        halo[:] = ring.unsqueeze(0).expand(self.layer_shape) * self.halo_alpha
        return halo

    def _make_streaks(self, cx: float, cy: float) -> Tensor:
        streaks = zeros(self.layer_shape, dtype=self.dtype, device=self.device)

        for i in range(self.streak_count):
            angle = (i / self.streak_count) * 3.1415926535
            dx = math.cos(angle)
            dy = math.sin(angle)
            
            proj = (self.proj_x - cx) * dx + (self.proj_y - cy) * dy
            perp = (self.proj_x - cx) * (-dy) + (self.proj_y - cy) * dx
            
            length_px = self.streak_length * self.smaller_side
            streak_width = 1.5
            
            length_mask = (1.0 - proj.abs() / length_px).clamp(0.0, 1.0) ** 2
            width_mask = (-(perp ** 2) / (2 * streak_width ** 2)).exp()
            
            streak = length_mask * width_mask * self.streak_alpha
            streaks[:] += streak.unsqueeze(0).expand(self.layer_shape)
        
        return streaks.clamp(0.0, 1.0)

    def _make_glow(self, cx: float, cy: float, alpha: float) -> Tensor:
        glow = zeros(self.layer_shape, dtype=self.dtype, device=self.device)
                
        dist = ((self.proj_x - cx)**2 + (self.proj_y - cy)**2).sqrt()
        r_px = self.glow_radius * self.smaller_side
        soft = (-(dist ** 2) / (2 * r_px ** 2)).exp()
        glow[:] = soft.unsqueeze(0).expand(self.layer_shape) * alpha
        return glow

    def _build_projections(self, H: int, W: int) -> None:
        self.proj_y = linspace(0, H-1, H, self.dtype, self.device)
        self.proj_y = self.proj_y.reshape((H, 1)).expand((H, W))
        self.proj_x = linspace(0, W-1, W, self.dtype, self.device)
        self.proj_x = self.proj_x.reshape((1, W)).expand((H, W))

    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        self.device = input.device
        self.dtype = input.dtype
        
        B, C, H, W = input.shape
        max_value = input.max().item()
        
        self._build_projections(H, W)
        self.layer_shape = (3, H, W)
        self.smaller_side = min(H, W)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for b in range(B):
                flare = zeros((3, H, W), dtype=self.dtype, device=self.device)
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

    def forward(self, input: TransformInput) -> TransformInput:
        B, C, H, W = input.image.shape
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
        scale: int = 100,
        octaves: int = 4,
        intensity_range: tuple[float, float] = (0.3, 0.7),
        fog_color: tuple[int, int, int] = (255, 255, 255)
    ) -> None:
        super().__init__()
        self.scale = scale
        self.octaves = octaves
        self.intensity_range = intensity_range
        self.fog_color = fog_color
        
    def _perlin_approx(
        self, 
        H: int, 
        W: int,
        device: Literal['cpu', 'cuda'],
        dtype: DTypeLike
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
        return (noise - min_value) / (noise.max().item() - min_value)

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
        return (out / out.max().item() * max_value).clamp(0.0, max_value)

    def forward(self, input: TransformInput) -> TransformInput:
        self._intensity = self._random_in_range(self.intensity_range)
        self._seed = int(self._random_in_range((0.0, 999999999.0)))
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )
        
class RandomRain(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

class RandomSnow(Transform):
    def __init__(
        self,
        brighness_coef: float = 1.5,
        snow_point_range: tuple[float, float] = (0.1, 0.3)
    ) -> None:
        super().__init__()
        self.brightness_coef = brighness_coef
        self.snow_point_range = snow_point_range
    
    def _transform(self, input: Tensor | None) -> Tensor | None:
        if input is None: return input
        max_value = input.max().item()
        norm = input / max_value
        
        r, g, b = norm.unbind(dim=1)
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b).unsqueeze(1)
        
        snow_mask = ((luminance-self._snow_point) / (1.0-self._snow_point))
        snow_mask = snow_mask.clamp(0.0, 1.0)
        
        noise = rand(snow_mask.shape, self._seed, input.dtype, input.device)
        snow_mask = (snow_mask + noise * 0.1).clamp(0.0, 1.0)
        
        out = norm + snow_mask * (1.0 - norm) * self.brightness_coef
        return (out * max_value).clamp(0.0, max_value)
    
    def forward(self, input: TransformInput) -> TransformInput:
        self._snow_point = self._random_in_range(self.snow_point_range)
        self._seed = int(self._random_in_range((0.0, 999999999.0)))
        return TransformInput(
            image     = self._transform(input.image),
            image2    = self._transform(input.image2),
            mask      = input.mask,
            boxes     = input.boxes,
            keypoints = input.keypoints
        )

class RandomShadow(Transform[Tensor, Tensor]):
    def __init__(self) -> None:
        raise NotImplementedError
        super().__init__()
    
    def forward(self, input: Tensor) -> Tensor:
        pass

