# Defines common utilities used across multiple examples.

import os
from   pathlib import Path

import nectarml
from   nectarml.vision import utils

###############################################################################
# OUTPUT
###############################################################################

def build_output_dir(path: str, allow_existing: bool = False) -> Path:
    '''Builds an directory for training outputs (checkpoints, examples).
    
    Args:
        path           : The path to build the output directory at.
        allow_existing : If False, this function will throw and exception if an
                         existing directory is found at the provided output 
                         path. Otherwise, it will just return a pathlib.Path to
                         the existing directory.
    
    Returns:
        Path : The path to the new or existing output directory.
        
    Raises:
        FileNotFoundError : If the parent directory of the output directory to
                            create does not exist.
    '''
    assert path != '', 'Please set OUTPUT_DIRECTORY to begin training.'
    
    path = Path(path).resolve()
    if not path.parent.exists():
        raise FileNotFoundError(
            f'Unable to build output directory at path: {path.as_posix()}\n'
            f'Parent directory does not exist.')
    
    path.mkdir(exist_ok=allow_existing)
    return path

def build_examples_dir(
    output_path:    Path, 
    allow_existing: bool = False
) -> Path:
    '''Builds subdirectory for example output images.
    
    Args:
        output_path    : The path to the root output directory.
        allow_existing : If False, this function will throw and exception if an
                         existing directory is found at the provided output 
                         path. Otherwise, it will just return a pathlib.Path to
                         the existing directory.
    
    Returns:
        Path : The path to the newly created example directory.
    '''
    examples_directory = Path(output_path, 'examples').resolve()
    examples_directory.mkdir(exist_ok=allow_existing)
    return examples_directory

def save_xyz_examples(
    model:       nectarml.nn.Module, 
    dataset:     nectarml.utils.data.Dataset,
    output_path: Path,
    epoch:       int
) -> None:
    '''Saves input, target, and inference example images to disk.

    Args:
        model       : The model to use for inference.
        dataset     : The dataset to draw the test data from.
        output_path : The system path to the directory to save the images to.
        epoch       : The epoch when the example images are being saved. Used
                      as a tag in the output file names.
    '''
    model.eval()

    device = model.parameters()[0].device
    with nectarml.no_grad():
        idx    = nectarml.random.RNG.randint(0, len(dataset)-1)
        x, y   = [t.unsqueeze(0).to(device) for t in dataset[idx]]
        y_fake = model(x)
        
        for item in [(x, 'A_real'), (y, 'B_real'), (y_fake, 'B_fake')]:
            image_path = Path(output_path, f'epoch{epoch}_{item[1]}.jpg')
            utils.save_image(item[0], image_path, normalize=True)
        
    model.train()

###############################################################################
# CONSOLE
###############################################################################

def print_losses(
    iteration:   int, 
    losses:      dict[str, float],
    losses_prev: dict[str, float] | None = None,
    precision:   int = 5
) -> None:
    '''Prints losses to the console with optional coloring.

    Args:
        iteration   : The iteration number that produced the losses being 
                      printed.
        losses      : A dict[str, float] containing the loss names as the keys, 
                      and the corresponding loss values to print.
        losses_prev : A dict[str, float] with the same format as `losses`, but
                      containing the previous timestep's losses rather than the 
                      current. If provided, this will be used to color the 
                      printed values green if the loss has decreased from the 
                      previous timestep, or red if it has increased. If 
                      `losses_prev` is None, all printouts will be white.
        precision   : The number of digits after the decimal place to round the
                      printed values to.
    '''
    output = (
        f'{'='*os.get_terminal_size()[0]}\n'
        f'Iteration: {iteration}\n'
        f'Loss:\n')
    
    for loss, curr in losses.items():
        if losses_prev is None: prev = curr
        else: prev = losses_prev[loss]
        
        ctag = ''
        if prev > 0.0:
            if curr < prev: ctag = '\u001b[38;2;0;255;0m'
            else:           ctag = '\u001b[38;2;255;0;0m'
        output += f'    {loss}: '
        output += ' '*(6-len(loss))+f'{ctag}{round(curr, precision)}\033[0m\n'
        
    print(output)

###############################################################################
# WEB VISUALIZER
###############################################################################

VISUALIZER: nectarml.viz.web.Client | None = None

def init_web_client(
    host: str = 'http://localhost',
    port: int = 8097
) -> None:
    '''Initializes a globally accessible web visualizer client.

    Args:
        host : The host name for the client to connect to.
        port : The port on the host to connect to.
    '''
    global VISUALIZER
    VISUALIZER = nectarml.viz.web.Client(host=host, port=port)

def clear_web_visualizer() -> None:
    '''Clears all images and graphs on the web visualizer dashboard.'''
    if VISUALIZER is None: return
    VISUALIZER.clear()

def update_web_images(
    tensors:      list[nectarml.Tensor],
    title:        str,
    window:       str = 'images',
    size:         int = 256,
    normalize:   bool = True
) -> None:
    '''Updates images on the web visualizer dashboard.

    Args:
        tensors   : A list containing the tensors to display as images.
        title     : The visible title for the image window on the visualizer.
        window    : The internal title of the window on the server to update.
        size      : The size to display each image at.
        normalize : If True, input tensors will be normalized to the (0-255)
                    uint8 value range automatically for display.
    '''
    if VISUALIZER is None: return
    VISUALIZER.images(
        tensors   = [i[0].detach().cpu() for i in tensors], 
        size      = size, 
        normalize = normalize,
        window    = window, 
        title     = title
    )

def update_web_graph(
    step:         float,
    losses:       list[float],
    legend:       list[str],
    window:       str,
    title:        str,
    v_axis_label: str = 'Value',
    h_axis_label: str = 'Epoch'
) -> None:
    '''Updates a graph on the web visualizer.

    Args:
        step         : The location to graph the data on the horizontal axis.
        losses       : A list containing the loss values to add to the graph.
        legend       : A list of strings defining the names to assign to each 
                       graphed loss. The order of these names correspond to 
                       that of `losses`.
        title        : The visible title for the graph window.
        window       : The internal title of the window to update.
        v_axis_label : The label to assign to the vertical axis of the graph.
        h_axis_label : The label to assign to the horizontal axis of the graph.
        
    '''
    if VISUALIZER is None: return
    VISUALIZER.line(
        X            = step,
        Y            = losses,
        legend       = legend,
        window       = window, 
        title        = title, 
        v_axis_label = v_axis_label, 
        h_axis_label = h_axis_label
    )
