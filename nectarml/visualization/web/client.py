import base64
import io
import json
import urllib.request
from typing import Any, Literal
from collections.abc import Iterable

import numpy as np

from nectarml.core import Tensor
from nectarml.vision.transforms import ToPIL, Resample

class Viz:
    def __init__(
        self, 
        host: str = 'localhost', 
        port: int = 8097
    ) -> None:
        self.url = f'http://{host}:{port}/push'

    def _post(
        self, 
        payload: dict[str, Any]
    ) -> None:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url, data=data,
            headers={'Content-Type': 'application/json'}
        )
        urllib.request.urlopen(req, timeout=2)

    def clear(self):
        self._post({'type': 'clear'})

    def image(
        self, 
        tensor: Tensor, 
        size: int | tuple[int, int],
        sampling_mode: Literal[
            'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        ] = 'nearest',
        preserve_aspect_ratio: bool = True,
        window: str = 'image', 
        title: str = '', 
        opts: dict[str, Any] | None = None
    ) -> None:
        resample = Resample(
            size=size, mode=sampling_mode, 
            preserve_aspect_ratio=preserve_aspect_ratio)
        img = ToPIL()(resample(tensor))
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png_b64 = base64.b64encode(buf.getvalue()).decode()
        self._post({
            'type': 'image', 'win': window,
            'title': title, 'data': png_b64,
            'opts': opts or {}
        })

    def images(
        self, 
        tensors: Iterable[Tensor], 
        size: int | tuple[int, int],
        nrow: int = 8, 
        sampling_mode: Literal[
            'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        ] = 'nearest',
        preserve_aspect_ratio: bool = True,
        window: str = 'images', 
        title: str = '', 
        opts: dict[str, Any] | None = None
    ) -> None:
        n = len(tensors)
        ncol = nrow
        nrows_grid = (n + ncol - 1) // ncol
        h, w = tensors[0].shape[:2]
        pad = 2
        grid = np.ones(
            (nrows_grid*(h+pad)-pad, ncol*(w+pad)-pad, 3), dtype='uint8') * 200
        for i, t in enumerate(tensors):
            r, c = divmod(i, ncol)
            y0, x0 = r*(h+pad), c*(w+pad)
            grid[y0:y0+h, x0:x0+w] = np.asarray(t)[:, :, :3]
    
        self.image(grid, size, sampling_mode=sampling_mode,
                   preserve_aspect_ratio=preserve_aspect_ratio,
                   window=window, title=title, opts=opts)

    def line(
        self, 
        Y: Iterable[float | int], 
        X: Iterable[float | int] | None = None, 
        window: str = 'plot', 
        title: str = '', 
        v_axis_label: str = '',
        h_axis_label: str = '',
        update: bool = False, 
        opts: dict[str, Any] | None = None
    ) -> None:
        if isinstance(Y[0], (list, tuple, np.ndarray)):
            series = [list(np.atleast_1d(s)) for s in Y]
        else:
            if X is not None and len(X) == 1 and len(Y) > 1:
                series = [[float(v)] for v in Y]
            else:
                series = [[float(v) for v in Y]]

        if X is None:
            X = list(range(len(series[0])))

        self._post({
            'type': 'line_update' if update else 'line',
            'win': window, 'title': title,
            'X': list(X), 'Y': series,
            'v_axis_label': v_axis_label, 'h_axis_label': h_axis_label,
            'opts': opts or {}
        })



