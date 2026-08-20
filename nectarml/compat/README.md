# This is mostly a placeholder module currently.

In the future, this will house compatibility modules for other frameworks. Currently, this module only supports conversion of tensors between NectarML and PyTorch. This can be done like so:

### NectarML -> PyTorch:
```python
import nectarml
from   nectarml.compat import pytorch

x = nectarml.rand((1, 1, 4, 4))
y = pytorch.tensor_nectar2torch(x)
```
### PyTorch -> NectarML:
```python
import torch
from   nectarml.compat import pytorch

x = torch.rand((1, 1, 4, 4))
y = pytorch.tensor_torch2nectar(x)
```
**Conversion will preserve:**
- Data
- DType
- Device
- Whether the tensor requires grad

***PLEASE NOTE:*** Conversion between NectarML and PyTorch requires that PyTorch be installed in the environment from which the conversion function is called. Please see [here](https://pytorch.org/get-started/locally/) for information on how to install PyTorch.

**More compatibility updates will be added in a future release.**