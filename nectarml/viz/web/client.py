import base64
import io
import json
import urllib.request
from typing          import Any, Literal
from collections.abc import Iterable

import numpy as np

from nectarml.core              import Tensor
from nectarml.vision.transforms import \
    ToPIL, Resample, MakeGrid, MinMaxNormalize

class Client:
    def __init__(
        self, 
        host: str = 'http://localhost', 
        port: int = 8097
    ) -> None:
        '''Initializes a web visualizer client instance.

        This client instance can interact with the NectarML web visualization
        server in various ways to simplify the process of visualizing data
        during training. 

        A web viz server instance must be running for the
        client to connect to, otherwise the client will raise an exception on
        init. A web viz server can be started by running the following command
        from the repository root: `python -m nectarml.viz.web`.

        ---
        **Please note: This feature is currently in the early beta stage. It is
        fully functional, but you may encounter bugs or things that lack polish
        while development continues.**
        ---
        
        Args:
            host : The host adress for the web visualzation server.
            port : The port on the host to open for the web server.
        '''
        self.url = f'{host}:{port}/push'
        self._post({'type': 'connection'})

    def _post(
        self, 
        payload: dict[str, Any]
    ) -> None:
        '''Posts to server.
        
        Arg:
            payload : The payload to send in the post request.
            
        Raises:
            ConnectionError : If unable to connect to web visualizer server.
        '''
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
        '''Abstract utility. Takes tensor, converts to image, posts to server.
        
        Args:
            tensor : The tensor to display
            window : The (internal) name of the window to display the image in.
            title  : Optional human readable title of the window for the image.
                     Will be visible on the image pane.
            opts   : Dictionary of options to send with the request, or None
                     if no options required.
            
        '''
        buf = io.BytesIO()
        ToPIL(normalize=False)(tensor).save(buf, format='JPEG')
        png_b64 = base64.b64encode(buf.getvalue()).decode()
        self._post({
            'type': 'image', 'win': window,
            'title': title, 'data': png_b64,
            'opts': opts or {}
        })

    def clear(self) -> None:
        '''Posts to server telling it to clear and reset all windows.'''
        self._post({'type': 'clear'})

    def image(
        self, 
        tensor:            Tensor, 
        size:              int | tuple[int, int],
        normalize:         bool = False,
        sampling_mode: Literal['nearest', 'bilinear', 'bicubic'] = 'nearest',
        keep_aspect_ratio: bool = True,
        window:            str  = 'image', 
        title:             str  = '', 
        opts:              dict[str, Any] | None = None
    ) -> None:
        '''Sends an image to the visualizer server for display.
        
        Args:
            tensor            : The tensor to display as an image.
            
            size              : The size to display the image at. Can either be 
                                a single integer, in which case that value will 
                                be treated as width and height, or a tuple of 
                                two integers for (W, H).
            
            normalize         : If True, the input tensor will be min/max 
                                normalized [0:1], then multiplied by 255 to 
                                fully saturate the uint8 range.
            
            sampling_mode     : The resampling mode to use when scaling the 
                                image to the desired size. Options are 
                                [`nearest`, `bilinear`, `bicubic`].
            
            keep_aspect_ratio : If True, the scaling will preserve the
                                aspect ratio of the input, scaled based on
                                the larger side.
            
            window            : The (internal) name of the window to 
                                display the image in.
            
            title             : Optional human readable title of the window for 
                                the image. Will be visible on the image pane.
            
            opts              : Dictionary of options to send with the request, 
                                or None if no options required.

        '''
        if normalize: tensor = MinMaxNormalize()(tensor) * 255
        resample = Resample(
            size=size, mode=sampling_mode, 
            preserve_aspect_ratio=keep_aspect_ratio)
        self._post_image(resample(tensor), window, title, opts)
        
    def images(
        self, 
        tensors:           Iterable[Tensor], 
        size:              int | tuple[int, int],
        nrow:              int  = 8, 
        padding:           int  = 2,
        normalize:         bool = False,
        sampling_mode: Literal['nearest', 'bilinear', 'bicubic'] = 'nearest',
        keep_aspect_ratio: bool = True,
        window:            str  = 'images', 
        title:             str  = '', 
        opts:              dict[str, Any] | None = None
    ) -> None:
        '''Displays a list of tensors as an image grid.
        
        Args:
            tensors           : The tensor to display.
            
            size              : The size to display **each** image at. Can 
                                either be a single integer, in which case that 
                                value will be treated as width and height, or a
                                tuple of two integers for (W, H).
            
            nrow              : The number of images to display per row before
                                wrapping.
                                
            padding           : The padding width (in pixels) to apply to all
                                side of each image.
            
            normalize         : If True, the input tensor will be each be
                                individually min/max normalized [0:1], then
                                multiplied by 255 to fully saturate the uint8 
                                range.
            
            sampling_mode     : The resampling mode to use when scaling the 
                                image to the desired size. Options are 
                                [`nearest`, `bilinear`, `bicubic`].
            
            keep_aspect_ratio : If True, the scaling will preserve the
                                aspect ratio of each input image, scaled based
                                on the larger side.
            
            window            : The (internal) name of the window to 
                                display the image in.
            
            title             : Optional human readable title of the window for 
                                the image. Will be visible on the image pane.
            
            opts              : Dictionary of options to send with the request, 
                                or None if no options required.
        '''
        tensors = [x.unsqueeze(0) if x.ndim == 3 else x for x in tensors]
        resample = Resample(
            size=size, mode=sampling_mode, 
            preserve_aspect_ratio=keep_aspect_ratio)
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
        '''Displays or updates a loss graph in the web visualizer.
        
        Args:
            Y            : A list containing the Y-axis values to display on 
                           the graph.
            X            : A single float or int, or an Iterable of floats or 
                           ints (length of X Yterable should match length of Y
                           Iterable) denoting the X-axis location to graph the
                           Y_values. Can also be Nonetype, in which case, the 
                           X-axis location will be inferred from Y-axis data.
            legend       : A list of string containing the names to assign to 
                           each line. The order of the legend list should
                           correspond to the order of the data in the Y 
                           Iterable.
            window       : The (internal) name of the graph pane to graph plot 
                           the data on.
            title        : Optional human readable title of the graph pane. 
                           Will be visible on the image pane.
            v_axis_label : Optional label for the vertical axis.
            h_axis_label : Optional label for the horizontal axis.
            update       : If True, the post request will trigger a graph 
                           update, plotting the data immediately. If not, the
                           graph will not be updated, allowing you to manually
                           trigger an update at a point of your choosing.
            opts         : Dictionary of options to send with the request, or 
                           None if no options required.
        '''
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
    

