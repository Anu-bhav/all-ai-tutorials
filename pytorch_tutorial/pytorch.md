# Pytorch Tutorial

Tutorial is based on this youtube playlist [PyTorch Tutorials - Complete Beginner](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)

# PyTorch Tutorial: [Tensor Basics](./Tensor%20Basics.ipynb)

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

# PyTorch Tutorial: [Gradient Calculation With Autograd](./Gradient%20Calculation%20With%20Autograd.ipynb)

This Jupyter Notebook (`Gradient Calculation With Autograd.ipynb`) explains and demonstrates PyTorch's automatic differentiation package: **Autograd**. Understanding Autograd is crucial for training neural networks as it handles the computation of gradients (backpropagation) automatically.

This notebook is part of a tutorial series based on the PyTorch Tutorials - Complete Beginner YouTube playlist.

## Key Concepts Covered:

1.  **`requires_grad` Attribute:**

    - How to create tensors that track computational history for gradient calculation (`requires_grad=True`).
    - Understanding that operations involving tensors with `requires_grad=True` build a computation graph.

2.  **Computation Graph:**

    - A brief conceptual overview of how PyTorch dynamically builds a graph representing the operations performed on tensors.
    - Identifying leaf nodes and intermediate nodes.

3.  **Gradient Calculation with `backward()`:**

    - Using the `.backward()` method on a scalar output (e.g., loss) to trigger gradient computation throughout the graph.
    - Understanding that `backward()` computes gradients using the chain rule.
    - The requirement for the output tensor to be scalar or providing a `gradient` argument to `backward()` for non-scalar outputs.

4.  **Accessing Gradients:**

    - How computed gradients are accumulated in the `.grad` attribute of the leaf tensors that had `requires_grad=True`.

5.  **Disabling Gradient Tracking:**

    - Using `torch.no_grad()` context manager: Ideal for inference or code blocks where gradients are not needed, saving memory and computation.
    - Using `.detach()`: Creates a new tensor that shares the same data but is detached from the computation history, preventing gradients from flowing back through it.

6.  **Gradient Accumulation and Zeroing:**
    - Understanding that gradients accumulate by default when `backward()` is called multiple times.
    - The necessity of zeroing gradients (e.g., using `optimizer.zero_grad()` or `tensor.grad.zero_()`) before each optimization step in a typical training loop.

## How to Use:

1.  Ensure you have PyTorch installed (`pip install torch`).
2.  Open `Gradient Calculation With Autograd.ipynb` in a Jupyter environment (like Jupyter Lab or VS Code with the Python extension).
3.  Run the cells sequentially to understand how PyTorch tracks operations and computes gradients automatically.

This notebook provides essential knowledge for understanding the mechanics behind training models in PyTorch.

# PyTorch Tutorial: [Backpropagation Explained](./Backpropagation.ipynb)

This Jupyter Notebook (`Backpropagation.ipynb`) provides a conceptual overview and a practical, simplified demonstration of the **backpropagation** algorithm using PyTorch. It illustrates how gradients are calculated with respect to model parameters (weights) and how these gradients are used to update the parameters to minimize a loss function.

This notebook builds upon the concepts introduced in `Gradient Calculation With Autograd.ipynb`.

## Key Concepts Covered:

1.  **Conceptual Foundation:**

    - **Chain Rule:** The mathematical foundation for calculating gradients through nested functions.
    - **Computational Graph:** Visualizing the sequence of operations and dependencies for gradient calculation.
    - **Three Key Steps:**
      1.  **Forward Pass:** Compute the model's output and the loss.
      2.  **Compute Local Gradients:** (Handled automatically by Autograd).
      3.  **Backward Pass:** Compute the gradient of the loss with respect to the weights using the chain rule.

2.  **Practical Implementation (Simplified):**
    - **Setting up Tensors:** Defining input (`x`), target (`y`), and weight (`w`) tensors. Crucially, setting `requires_grad=True` for the weight tensor (`w`) to track operations and enable gradient calculation for it.
    - **Forward Pass:** Performing calculations to get the model's prediction (`y_hat`) and computing the loss (e.g., Mean Squared Error: `(y_hat - y)**2`).
    - **Backward Pass (`loss.backward()`):** Initiating the backpropagation process. PyTorch's Autograd engine calculates the gradients (e.g., `dLoss/dw`) automatically based on the recorded computation graph.
    - **Accessing Gradients:** Retrieving the computed gradient for the weight using `w.grad`.
    - **Manual Weight Update:** Implementing a basic gradient descent step: `weight = weight - learning_rate * gradient`. This is done within a `torch.no_grad()` block to prevent this update step from being tracked in the computation graph.
    - **Zeroing Gradients (`w.grad.zero_()`):** Resetting the gradients after updating the weights. This is essential because gradients accumulate by default on subsequent `.backward()` calls.

## How to Use:

1.  Ensure you have PyTorch installed (`pip install torch`).
2.  Open `Backpropagation.ipynb` in a Jupyter environment (like Jupyter Lab or VS Code with the Python extension).
3.  Run the cells sequentially to follow the conceptual explanation and the step-by-step implementation of a single backpropagation and weight update cycle.

This notebook provides a foundational understanding of how automatic differentiation (Autograd) is used to implement the backpropagation algorithm, which is the core mechanism for training most neural networks. While this example uses manual weight updates, in practice, PyTorch's `torch.optim` module provides optimized algorithms for this step.
