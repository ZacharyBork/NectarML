# NectarML

**A lightweight, numpy-based machine learning library.**

# TO-DO

### ✅  Tensor Class

### ✅  Tensor Creation Library

### ✅  Typing Abstraction Layer

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
    ✅ 10. Padding
    ✅ 11. Reductions
    ✅ 12. Shapes
    ✅ 13. Interpolation

### ✅  Normalization Modules (nectarml.nn.norm) 

### ✅  Activation Modules (nectarml.nn.activation)

### ✅  Loss Modules (nectarml.nn.loss)

### ✅  Weight Initialization (nectarml.nn.init)

### 🟡  Layers:

    ✅ 1. Linear
    ✅ 2. Upsample
    ✅ 3. Identity
    ❌ 4. Attention
    ❌ 5. Convolutions
        ❌ 1. Conv1d
        ❌ 2. Conv2d
        ❌ 3. Conv3d
        ❌ 4. ConvTranspose1d
        ❌ 5. ConvTranspose2d
        ❌ 6. ConvTranspose3d
    ❌ 6. Pooling
        ❌ 1. MaxPool1d
        ❌ 2. MaxPool2d
        ❌ 3. MaxPool3d
        ❌ 4. AvgPool1d
        ❌ 5. AvgPool2d
        ❌ 6. AvgPool3d

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

### ✅  Dataset

### 🟡  Dataloader

### ✅  Image I/O Utilities

### 🟡  Image Transform/Augmentation Utilities

    🟡 1. Spatial / Geometric
    🟡 2. Color / Photometric
    ❌ 3. Blur / Noise
    ❌ 4. Dropout / Erasing
    ❌ 5. Normalization
    ❌ 6. Format / Type
    ✅ 7. Composition

### 🟡  CUDA:

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

### 🟡  Other:

    ✅ 1. Autocast Context
    ✅ 2. Nograd Context/Decorator
    ❌ 3. AMP flags for functions/methods

### 🟡  Documentation:

    🟡 1. Class/Function/Method Docstrings
    ❌ 2. Markdown Documentation
    ❌ 3. README.md


