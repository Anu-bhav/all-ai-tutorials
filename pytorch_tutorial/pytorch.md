# Pytorch Tutorial

Tutorial is based on this youtube playlist [PyTorch Tutorials - Complete Beginner](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)

1. Basics

# PyTorch Tutorial: [Tensor Basics](Tensor Basics.ipynb)

This Jupyter Notebook (`Tensor Basics.ipynb`) serves as an introduction to the fundamental building block of PyTorch: the **Tensor**. It covers the creation, manipulation, and basic operations associated with PyTorch tensors.

This notebook is part of a tutorial series based on the PyTorch Tutorials - Complete Beginner YouTube playlist.

## Key Concepts Covered:

1.  **Tensor Creation:**

    - `torch.empty()`: Creates an uninitialized tensor.
    - `torch.rand()`: Creates a tensor with random values uniformly distributed between 0 and 1.
    - `torch.randn()`: Creates a tensor with random values from a standard normal distribution (mean 0, variance 1).
    - `torch.zeros()`: Creates a tensor filled with zeros.
    - `torch.ones()`: Creates a tensor filled with ones.
    - `torch.tensor()`: Creates a tensor directly from data (e.g., Python lists).
    - Specifying `dtype` (e.g., `torch.int`, `torch.float`, `torch.double`) during creation.

2.  **Tensor Attributes:**

    - `x.dtype`: Checking the data type of a tensor.
    - `x.size()`: Getting the dimensions (shape) of a tensor.

3.  **Tensor Operations:**

    - Basic arithmetic: Addition (`+`, `torch.add`), subtraction, multiplication, division.
    - In-place operations: Functions ending with `_` (e.g., `y.add_(x)`) modify the tensor directly.

4.  **Indexing and Slicing:**

    - Accessing specific elements or sub-tensors using standard Python slicing syntax (e.g., `x[0]`, `x[:, 0]`, `x[0, 0]`).
    - Using `.item()` to get the Python number value from a single-element tensor.

5.  **Reshaping Tensors:**

    - `x.view()`: Reshaping a tensor without changing its data. Using `-1` to infer a dimension.

6.  **NumPy Bridge:**

    - Converting a PyTorch Tensor to a NumPy array: `a.numpy()`.
    - Converting a NumPy array to a PyTorch Tensor: `torch.from_numpy(a)`.
    - **Important:** When tensors are on the CPU, the PyTorch tensor and the NumPy array share the same underlying memory. Modifying one will modify the other.

7.  **GPU / Device Handling:**
    - Checking for GPU availability: `torch.cuda.is_available()`.
    - Creating a device object: `torch.device("cuda")` or `torch.device("cpu")`.
    - Creating tensors directly on a specific device: `torch.ones(..., device=device)`.
    - Moving existing tensors to a device: `x.to(device)`.
    - Performing operations on tensors located on the GPU.
    - Moving tensors back to the CPU before converting to NumPy: `z.to("cpu").numpy()`.

## How to Use:

1.  Ensure you have PyTorch and NumPy installed (`pip install torch numpy`).
2.  If you have an NVIDIA GPU, ensure CUDA drivers and the correct PyTorch version with CUDA support are installed.
3.  Open `Tensor Basics.ipynb` in a Jupyter environment (like Jupyter Lab or VS Code with the Python extension).
4.  Run the cells sequentially to understand each concept.

This notebook provides a solid foundation for working with tensors, which is essential for any further work in PyTorch.
