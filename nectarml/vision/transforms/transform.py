from __future__ import annotations

from typing          import TypeVar, Generic
from collections.abc import Iterable

from nectarml.core   import Tensor
from nectarml.random import RNG
from nectarml.vision.transforms.common import TransformInput

TInputType  = TypeVar('TInputType')
TOutputType = TypeVar('TOutputType')
            
class Transform:
    def __init__(self: Transform, p: float = 1.0) -> None:
        '''Initializes a base Transform class.

        This is meant to serve as a parent class for all vision transform
        classes. It implements a `__call__` method which handles a few things:

        1. Packing of input arguments into a TransformInput utility class, and
           unpacking them to return as outputs. This allows every transform
           class to just accept the same standardized inputs and also handles
           input sanitization automatically.
        2. Random chance of applying a given transform. Every time `__call__` 
           is envoked, the parent automatically does a random check against the
           transform's probability value and bypasses accordingly.
        3. Calls the transform's `forward` method (if not skipped by 
           probability). This is the core of how the transform classes work.
           Each one implements a forward class which processes inputs and then
           packs them back in to `TransformInput` objects for the next 
           transform.

        The base Transform class also stores a reference to the global random
        number generator, and implements its own random value utility called
        `_random_in_range`. This utility takes a two item tuple of integers
        or floats and generates a random value in between the two (inclusive at
        both ends). This is used to simplify the process of having transforms
        accept value ranges to pick a random value each time the transform is
        called, which is used all over the place.
        
        And finally, the base transform stores a global epsilon value for
        any division operations the transform needs to do, just as a 
        convenience to avoid division by zero errors. It can be accessed with
        `Transform._epsilon` for read and write, and defaults to `1e-8`.

        Args:
            p : The probability of the transform being applied to any given
                input.
        '''
        self.rng      = RNG
        self.p        = p
        self._epsilon = 1e-8
    
    ### UTILS ###
    
    def _random_in_range(
        self:        Transform, 
        value_range: tuple[int | float, int | float] = (0.0, 1.0)
    ) -> float:
        '''Generates a random value in the `value_range`.

        The range is inclusive on both ends. The random value is generated in
        the context of the global random state, so it is affected by manual
        seeding.

        Args:
            value_range : A tuple containing the (min, max) values for the 
                          range.
        
        Returns:
            float : The generated random value (always as a floating point
                    number).
        '''
        return self.rng.randfloat(value_range[0], value_range[1])
        
    ### FORWARD ###
    
    def forward(self: Transform, input: TransformInput) -> TransformInput:
        '''Implemented by the child class. Defines the logic for the transform.

        Must accept a TransformInput as its only argument, and return only a
        TransformInput. Apart from that, there really are no limits. You may
        process the items contained in the TransformInput however you wish.

        This method will be invoked automatically when the transform class is
        called (like so: `Transform(image=x, image2=y)`), assuming the
        automatic probability check does not bypass the transform on the given
        call.
        
        Args:
            input : The TransformInput for the tranform class to process.
            
        Returns:
            TransformInput : A TransformInput object containing the results of
                             the given transform.
        '''
        raise NotImplementedError
    
    def _call(self: Transform, input: TransformInput) -> TransformInput:
        '''Utility method to handle random transform bypassing.

        Generates a random value between 0 and 1, and checks if its greater
        than the transform's probability value. If so, it just returns the
        input. If not, it runs the transform's `forward()` method, passing it
        the `input` TransformInput, and returning the result.

        Args:
            input : The TransformInput for the tranform class to process.
            
        Returns:
            TransformInput : A TransformInput object containing the results of
                             the given transform.
        '''
        if    self.rng.random() > self.p: result = input
        else: result = self.forward(input)
        return result
    
    def __call__(
        self:     Transform, 
        *args:    Tensor | Iterable[Tensor], 
        **kwargs: dict[str, Tensor]
    ) -> Tensor | tuple[Tensor]:
        '''__call__ method for Transform class.

        Automatically packs input `args` and `kwargs` into a TransformInput
        object, checks the transform's probability, and runs the transform's
        forward method accordingly.

        `args` can be one of many things:
        1. A PIL.Image
        2. A numpy.ndarray
        3. A nectarml.Tensor
        4. A torch.Tensor (used for conversion tools)

        Or, technically, anything you want. They are passes to the transform 
        like so:
        ```
        Transform(arg1, arg2, arg3)
        ```
        With their positions defining their assignment in the TransformInput.
        The convention is as follows:
        - Single input : (image,)
        - Two Inputs   : (image, mask)
        - Three inputs : (image, image2, mask)

        This only works for up to three inputs. To use the boxes and keypoints
        of the TransformInput, you must instead use `kwargs`. This can be done
        like so:
        ```
        Transform(
            image     = arg1, 
            image2    = arg2, 
            mask      = arg3, 
            boxes     = arg4, 
            keypoints = arg5
        )
        ```
        You are not required to populate every input when using `kwargs`.

        Args:
            *args    : The input items for the TransformInput (see above).
            **kwargs : The input items by keyword for the TransformInput 
                       (see above).
        
        Returns:
            Tensor | tuple[Tensor] : The results of the transform, unpacked 
                                     from the TransformInput wrapper.
        
        '''
        input  = TransformInput.from_args(args, kwargs)
        result = self._call(input)
        return result.to_output(args, kwargs)
    
    ### INSPECTION ###
    
    def __repr__(self: Transform) -> str:
        '''__repr__ inspection method for transform classes.

        Returns:
            str : A string denoting the class type of the transform. Useful
                  for visualizing and inspecting transform stacks.
        '''
        return f'{self.__class__}'
    
class UtilityTransform(Transform, Generic[TInputType, TOutputType]):
    def __init__(self: UtilityTransform) -> None:
        '''Thin wrapper around base Transform to help with type hinting.

        By default, the call method of the base `Transform` class will treat
        all returns as Tensors. Which technically is not true for all 
        Transforms, as some of the utility ones return numpy arrays and 
        whatnot. This just allows for correct return type hinting using
        generics.

        This is a temporary stopgap, and will likely be replaced with a more
        permenant solution in the future.
        '''
        super().__init__()
        
    def __call__(self: UtilityTransform, input: TInputType) -> TOutputType:
        return self.forward(input)

