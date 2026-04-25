import base64
import io
import json
import urllib.request
from typing import Any, Literal
from collections.abc import Iterable

import numpy as np

from nectarml.core import Tensor
from nectarml.vision.transforms import \
    ToPIL, Resample, MakeGrid, MinMaxNormalize

class Client:
    def __init__(
        self, 
        host: str = 'http://localhost', 
        port: int = 8097
    ) -> None:
        self.url = f'{host}:{port}/push'
        self._post({'type': 'connection'})

    def _post(
        self, 
        payload: dict[str, Any]
    ) -> None:
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            self.url, data=data,
            headers={'Content-Type': 'application/json'}
        )
        try: urllib.request.urlopen(req, timeout=2)
        except urllib.error.URLError:
            raise ConnectionError(
                'Unable to connect to web visualizer server. Please ensure '
                'it is running.')
        
    def _post_image(
        self,
        tensor: Tensor,
        window: str = 'images', 
        title:  str = '', 
        opts:   dict[str, Any] | None = None
    ) -> None:
        buf = io.BytesIO()
        ToPIL(normalize=False)(tensor).save(buf, format='JPEG')
        png_b64 = base64.b64encode(buf.getvalue()).decode()
        self._post({
            'type': 'image', 'win': window,
            'title': title, 'data': png_b64,
            'opts': opts or {}
        })

    def clear(self):
        self._post({'type': 'clear'})

    def image(
        self, 
        tensor:        Tensor, 
        size:          int | tuple[int, int],
        normalize:     bool = False,
        sampling_mode: Literal[
            'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        ] = 'nearest',
        preserve_aspect_ratio: bool = True,
        window:                str  = 'image', 
        title:                 str  = '', 
        opts:                  dict[str, Any] | None = None
    ) -> None:
        if normalize: tensor = MinMaxNormalize()(tensor) * 255
        resample = Resample(
            size=size, mode=sampling_mode, 
            preserve_aspect_ratio=preserve_aspect_ratio)
        self._post_image(resample(tensor), window, title, opts)
        
    def images(
        self, 
        tensors:       Iterable[Tensor], 
        size:          int | tuple[int, int],
        nrow:          int  = 8, 
        padding:       int  = 2,
        normalize:     bool = False,
        sampling_mode: Literal[
            'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        ] = 'nearest',
        preserve_aspect_ratio: bool = True,
        window:                str  = 'images', 
        title:                 str  = '', 
        opts:                  dict[str, Any] | None = None
    ) -> None:
        tensors = [x.unsqueeze(0) if x.ndim == 3 else x for x in tensors]
        resample = Resample(
            size=size, mode=sampling_mode, 
            preserve_aspect_ratio=preserve_aspect_ratio)
        tensors = [resample(MinMaxNormalize()(x) * 255) 
                   if normalize else resample(x) for x in tensors ]

        grid = MakeGrid(nrow=nrow, padding=padding, pad_value=100)(tensors)
        self._post_image(grid, window, title, opts)

    def line(
        self, 
        Y:            Iterable[float | int], 
        X:            float | int | Iterable[float | int] | None = None, 
        legend:       list[str] = [],
        window:       str  = 'plot', 
        title:        str  = '', 
        v_axis_label: str  = '',
        h_axis_label: str  = '',
        update:       bool = True, 
        opts:         dict[str, Any] | None = None
    ) -> None:
        if isinstance(Y[0], list | tuple | np.ndarray):
            series = [list(np.atleast_1d(s)) for s in Y]
        else:
            if X is not None:
                if isinstance(X, int | float): X = [X]
                if len(X) == 1 and len(Y) > 1:
                      series = [[float(v)] for v in Y]
                else: series = [[float(v) for v in Y]]
            else: X = list(range(len(series[0])))

        self._post({
            'type':         'line_update' if update else 'line',
            'win':          window, 
            'title':        title,
            'X':            list(X),  
            'Y':            series,   
            'legend':       legend, 
            'v_axis_label': v_axis_label, 
            'h_axis_label': h_axis_label,
            'opts':         opts or {}
        })

def start_client(
    host: str = 'http://localhost', 
    port: int = 8097
) -> Client:
    '''Starts a web visualizer client at the given host and port.
    
    Args:
        host : The host for the web visualizer application.
        port : The port to open for the web visualizer application.
        
    Returns:
        Client : A web visualizer client instance connected to the given host
            and port, which can be used to update the web visualizer app.
    '''
    return Client(host, port)    
    

