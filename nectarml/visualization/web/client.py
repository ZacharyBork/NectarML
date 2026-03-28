import base64
import io
import json
import urllib.request
from typing import Any, Literal
from collections.abc import Iterable

import numpy as np

from nectarml.tensor import Tensor
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

    def image(
        self, 
        tensor: Tensor, 
        size: int | tuple[int, int],
        sampling_mode: Literal[
            'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        ] = 'nearest',
        win: str = 'image', 
        title: str = '', 
        opts: dict[str, Any] | None = None
    ) -> None:
        resampled = Resample(size=size, mode=sampling_mode)(tensor)
        img = ToPIL()(resampled)
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png_b64 = base64.b64encode(buf.getvalue()).decode()
        self._post({
            'type': 'image', 'win': win,
            'title': title, 'data': png_b64,
            'opts': opts or {}
        })

    def images(
        self, 
        tensors: Iterable[Tensor], 
        nrow: int = 8, 
        win: str = 'images', 
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
    
        self.image(grid, win=win, title=title, opts=opts)

    def line(
        self, 
        Y: int, 
        X: int | None = None, 
        win: str = 'plot', 
        title: str = '', 
        update: bool = False, 
        opts: dict[str, Any] | None = None
    ) -> None:
        if not isinstance(Y[0], (list, tuple)):
            Y = [Y]
        if X is None:
            X = list(range(len(Y[0])))
        opts = opts or {}
        self._post({
            'type': 'line_update' if update else 'line',
            'win': win, 'title': title,
            'X': list(X), 'Y': [list(s) for s in Y],
            'opts': opts
        })



