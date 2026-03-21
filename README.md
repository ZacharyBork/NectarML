# NectarML

**A lightweight, numpy-based machine learning library.**

# TO-DO

### ✅  Tensor Class

### ✅  Typing Abstraction Layer

### ✅  Tensor Creation Library

    ✅ 1. Creation/Duplication
        ✅ 1. Clone
        ✅ 2. Zeros Like
        ✅ 3. Ones Like
        ✅ 4. Rand Like
        ✅ 5. Full Like
        ✅ 6. Empty Like
    ✅ 2. Fixed Shape
        ✅ 1. Zeros
        ✅ 2. Ones
        ✅ 3. Rand
        ✅ 4. Randn
        ✅ 5. Full
        ✅ 6. Empty
        ✅ 7. Eye
        ✅ 8. Arange
        ✅ 9. Linspace

### ✅  CPU Library (nectarml.cpu)

    ✅ 1. Combinations
    🟡 2. Conv
        ✅ 1. Conv1d
        ✅ 2. Conv2d
        ❌ 3. Conv3d
        ❌ 4. ConvTranspose1d
        ❌ 5. ConvTranspose2d
        ❌ 6. ConvTranspose3d
    ✅ 3. Indexing
    ✅ 4. Masking
    ✅ 5. Math
    ✅ 6. Padding
    ✅ 7. Reductions
    ✅ 8. Shapes
    ✅ 9. Interpolation

### ✅  Functional Library

    ✅ 1.  Activation
    ✅ 2.  Attention
        ✅ 1. ScaledDotProductAttention
        ✅ 2. MultiheadAttention
    ✅ 3.  Combination
    🟡 4.  Conv
        ✅ 1. Conv1d
        ✅ 2. Conv2d
        ❌ 3. Conv3d
        ❌ 4. ConvTranspose1d
        ❌ 5. ConvTranspose2d
        ❌ 6. ConvTranspose3d
    ❌ 5.  Pooling
        ❌ 1. MaxPool1d
        ❌ 2. MaxPool2d
        ❌ 3. MaxPool3d
        ❌ 4. AvgPool1d
        ❌ 5. AvgPool2d
        ❌ 6. AvgPool3d
    ✅ 6.  Indexing
    ✅ 7.  Loss
    ✅ 8.  Math
    ✅ 9.  Normalization
        ✅ 1. BatchNorm1d
        ✅ 2. BatchNorm2d
        ✅ 3. BatchNorm3d
        ✅ 4. InstanceNorm1d
        ✅ 5. InstanceNorm2d
        ✅ 6. InstanceNorm3d
        ✅ 7. GroupNorm
        ✅ 8. LayerNorm
    ✅ 10. Padding
    ✅ 11. Reductions
    ✅ 12. Shapes
    ✅ 13. Interpolation
    ✅ 14. Dropout
        ✅ 1. Dropout
        ✅ 2. Alpha Dropout
        ✅ 3. Feature Alpha Dropout
        ✅ 4. Dropout1d
        ✅ 5. Dropout2d
        ✅ 6. Dropout3d

### ✅  Activation Modules (nectarml.nn.activation)

    ✅ 1.  ReLU
    ✅ 2.  LeakyReLU
    ✅ 3.  ELU
    ✅ 4.  SELU
    ✅ 5.  Sigmoid
    ✅ 6.  Tanh
    ✅ 7.  Softmax
    ✅ 8.  LogSoftmax
    ✅ 9.  GeLU
    ✅ 10. SiLU/Swish
    ✅ 11. Softplus
    ✅ 12. Mish
    ✅ 13. Hardtanh
    ✅ 14. Hardsigmoid
    ✅ 15. Hardswish
    ✅ 16. Softsign
    ✅ 17. Softmin

### ✅  Loss Modules (nectarml.nn.loss)

    ✅ 1. Regression
        ✅ 1. L1Loss/MAELoss
        ✅ 2. L2Loss/MSELoss
        ✅ 3. RMSELoss
        ✅ 4. HuberLoss
        ✅ 5. LogCoshLoss
    ✅ 2. Classification
        ✅ 1. BCELoss
        ✅ 2. CrossEntropyLoss
        ✅ 3. NLLLoss
        ✅ 4. HingeLoss
        ✅ 5. Hinge2Loss
    ✅ 3. Probabilistic
        ✅ 1. KLDivergenceLoss
        ✅ 2. BCEWithLogitsLoss
    ✅ 4. Ranking
        ✅ 1. TripletMarginLoss

### ✅  Weight Initialization (nectarml.nn.init)

    ✅ 1. Constant
        ✅ 1. Zeros
        ✅ 2. Ones
        ✅ 3. Constant
        ✅ 4. Eye
        ✅ 5. Dirac
    ✅ 2. Random
        ✅ 1. Uniform
        ✅ 2. Normal
    ✅ 3. Variance Scaling
        ✅ 1. Xavier Uniform
        ✅ 2. Xavier Normal
        ✅ 3. Kaiming Uniform
        ✅ 4. Kaiming Normal
    ✅ 4. Other
        ✅ 1. Trunc Normal
        ✅ 2. Orthogonal
        ✅ 3. Sparse

### ✅  Inspection Utilities

    ✅ 1. Is Infinite
    ✅ 1. Is Finite
    ✅ 1. Is NaN
    ✅ 1. Has Infinite
    ✅ 1. Has NaN

### ✅  Module Composition (nectarml.nn.composition)

    ✅ 1. ModuleDict
    ✅ 2. ModuleList
    ✅ 3. Sequential

### 🟡  Layers

    ✅ 1. Linear
    ✅ 2. Upsample
    ✅ 3. Identity
    ✅ 4. Normalization
        ✅ 1. BatchNorm1d/2d/3d
        ✅ 2. InstanceNorm1d/2d/3d
        ✅ 3. GroupNorm
        ✅ 4. LayerNorm
    ✅ 5. Padding
        ✅ 1. ConstantPad1d/2d/3d
        ✅ 2. ReflectionPad1d/2d/3d
        ✅ 3. ReplicationPad1d/2d/3d
        ✅ 4. CircularPad1d/2d/3d
        ✅ 5. ZeroPad1d/2d/3d
    ❌ 6. Attention
        ❌ 1. ScaledDotProductAttention
        ❌ 2. MultiheadAttention
    ❌ 7. Convolutions
        ❌ 1. Conv1d
        ❌ 2. Conv2d
        ❌ 3. Conv3d
        ❌ 4. ConvTranspose1d
        ❌ 5. ConvTranspose2d
        ❌ 6. ConvTranspose3d
    ❌ 8. Pooling
        ❌ 1. MaxPool1d
        ❌ 2. MaxPool2d
        ❌ 3. MaxPool3d
        ❌ 4. AvgPool1d
        ❌ 5. AvgPool2d
        ❌ 6. AvgPool3d
    ❌ 9. Dropout
        ❌ 1. Dropout
        ❌ 2. Alpha Dropout
        ❌ 3. Feature Alpha Dropout
        ❌ 4. Dropout1d
        ❌ 5. Dropout2d
        ❌ 6. Dropout3d

### 🟡  Optimizers

    ✅ Base Optimizer

    First-order:
        ✅ 1. SGD
        ✅ 2. SGD w/ Nesterov momentum
        ✅ 3. Adam
        ❌ 4. AdamW
        ❌ 5. NAdam
        ❌ 6. RAdam
        ❌ 7. Adamax
        ❌ 8. Adagrad
        ❌ 9. RMSprop
        
    Second-order:
        ❌ 1. Adadelta
        ❌ 2. ASGD
        ❌ 3. LBFGS

    Other:
        ❌ 1. AdaFactor
        ❌ 2. Lion
        ❌ 3. Sophia

### ✅  LR Schedulers

    ✅ 1.  Base Scheduler
    ✅ 2.  StepLR
    ✅ 3.  MultiStepLR
    ✅ 4.  ConstantLR
    ✅ 5.  LinearLR
    ✅ 6.  ExponentialLR
    ✅ 7.  PolynomialLR
    ✅ 8.  CosineAnnealingLR
    ✅ 9.  CosineAnnealingWarmRestarts
    ✅ 10. ReduceLROnPlateau
    ✅ 11. CyclicLR
    ✅ 12. OneCycleLR

### ✅  Data Utils

    🟡 1. Dataloader
    ✅ 2. Default Collate Function
    ✅ 3. Dataset
        ✅ 1. Base Dataset Classes
            ✅ 1. Base Dataset
            ✅ 2. Base IterableDataset
        ✅ 2. Core Datasets
            ✅ 3. ImageFolderDataset
            ✅ 4. TensorDataset
            ✅ 5. Subset
            ✅ 6. CSVDataset
        ✅ 3. Combined Datasets
            ✅ 7. ConcatDataset
            ✅ 8. ChainDataset
            ✅ 9. StackDataset
    ✅ 4. Samplers
        ✅ 1. Batch Sampler
        ✅ 2. Index Samplers
            ✅ 1. Sampler
            ✅ 2. SequentialSampler
            ✅ 3. RandomSampler
            ✅ 4. WeightedRandomSampler
            ✅ 5. SubsetRandomSampler

### ✅  Image I/O Utilities

    ✅ 1. PIL -> Tensor
    ✅ 2. Tensor -> PIL
    ✅ 3. Load Image
    ✅ 4. Save Image
    ✅ 5. Make Grid

### 🟡  Image Transform/Augmentation Utilities

    🟡 1. Spatial / Geometric
    🟡 2. Color / Photometric
    ❌ 3. Blur / Noise
    ❌ 4. Dropout / Erasing
    ❌ 5. Normalization
    ❌ 6. Format / Type
    ✅ 7. Composition

### 🟡  CUDA

    🟡 1. Host
        🟡 1. Conv
            ✅ 1. Conv1d
            ✅ 2. Conv2d
            ❌ 3. Conv3d
            ❌ 4. ConvTranspose1d
            ❌ 5. ConvTranspose2d
            ❌ 6. ConvTranspose3d
        ✅ 2.  Upsample
        ❌ 3.  Pooling
        ✅ 4.  Elementwise
        ✅ 5.  Indexing
        ✅ 6.  Matmul
        ✅ 7.  Memory
        ✅ 8.  Padding
        ✅ 9.  Reductions
        ✅ 10. Combination
        🟡 11. Vision
    🟡 2. Kernels
        🟡 1. Conv
            ✅ 1. Conv1d
            ✅ 2. Conv2d
            ❌ 3. Conv3d
            ❌ 4. ConvTranspose1d
            ❌ 5. ConvTranspose2d
            ❌ 6. ConvTranspose3d
        ✅ 2.  Upsample
        ❌ 3.  Pooling
        ✅ 4.  Elementwise
        ✅ 5.  Indexing
        ✅ 6.  Matmul
        ✅ 7.  Memory
        ✅ 8.  Padding
        ✅ 9.  Reductions
        ✅ 10. Combination
        🟡 11. Vision
    🟡 3. Bindings
    ✅ 4. CMakeLists
    ❌ 5. Organization and Cleanup

### 🟡  Other

    ✅ 1. Autocast Context
    ✅ 2. Nograd Context/Decorator
    ❌ 3. AMP flags for functions/methods

### 🟡  Compatibility Layersc

    🟡 1. PyTorch
    ❌ 2. ONNX
    ❌ 3. Jax

### 🟡  Documentation

    🟡 1. Class/Function/Method Docstrings
    ❌ 2. Markdown Documentation
    ❌ 3. README.md


