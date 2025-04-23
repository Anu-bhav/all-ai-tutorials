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

# PyTorch Tutorial: [Gradient Descent with Autograd and Backpropagation](./Gradient%20Descent%20with%20Autograd%20and%20Backpropagation.ipynb)

This Jupyter Notebook (`Gradient Descent with Autograd and Backpropagation.ipynb`) provides a practical demonstration of training a simple linear regression model using gradient descent. It serves two main purposes:

1.  Illustrates the complete process manually using **NumPy**, including:
    - Forward pass (prediction)
    - Loss calculation (Mean Squared Error - MSE)
    - Manual gradient computation
    - Weight update step
2.  Replicates the same process using **PyTorch**, highlighting the power of **Autograd** for automatic gradient calculation (backpropagation).

This notebook effectively contrasts the manual approach with PyTorch's automated capabilities, building upon concepts from `Tensor Basics.ipynb`, `Gradient Calculation With Autograd.ipynb`, and `Backpropagation.ipynb`.

## Key Concepts Covered:

1.  **Linear Regression Model:** Implementing a simple model `y = w * x`.
2.  **Loss Function:** Using Mean Squared Error (MSE) to quantify the difference between predictions and actual values.
3.  **Gradient Descent:** The iterative optimization algorithm used to minimize the loss by updating model parameters (weights).
4.  **Manual Implementation (NumPy):**
    - Defining model (`forward`), loss (`loss`), and gradient (`gradient`) functions explicitly.
    - Implementing the training loop with manual calculations for prediction, loss, gradient, and weight updates.
5.  **PyTorch Implementation:**
    - Setting up data and weights as PyTorch tensors.
    - Crucially, setting `requires_grad=True` on the weight tensor to enable gradient tracking.
    - Defining model (`forward`) and loss (`loss`) functions using PyTorch operations.
    - **Automatic Gradient Calculation:** Using `loss.backward()` to automatically compute gradients (dLoss/dw) via Autograd and backpropagation. No manual gradient function is needed.
    - **Accessing Gradients:** Using `w.grad` to retrieve the computed gradient.
    - **Weight Update:** Performing the update step `w -= learning_rate * w.grad` within a `torch.no_grad()` block to prevent tracking this operation in the computation graph.
    - **Zeroing Gradients:** Using `w.grad.zero_()` after each update to reset gradients for the next iteration, preventing accumulation.
6.  **Comparison:** Implicitly shows the simplification and efficiency gained by using PyTorch's Autograd compared to manual gradient derivation and implementation.

## How to Use:

1.  Ensure you have PyTorch and NumPy installed (`pip install torch numpy`).
2.  Open `Gradient Descent with Autograd and Backpropagation.ipynb` in a Jupyter environment (like Jupyter Lab or VS Code with the Python extension).
3.  Run the cells sequentially. First, observe the manual NumPy implementation, then see how the same result is achieved more easily with PyTorch and Autograd.

This notebook provides a concrete example of a basic machine learning training loop and clearly demonstrates the practical advantage of using automatic differentiation frameworks like PyTorch.

# PyTorch Tutorial: [Training Pipeline - Model, Loss, and Optimizer](./Training%20Pipeline%20-%20Model,%20Loss,%20and%20Optimizer.ipynb)

This Jupyter Notebook (`Training Pipeline - Model, Loss, and Optimizer.ipynb`) demonstrates the standard PyTorch training pipeline by building and training a simple linear regression model. It introduces key PyTorch components for streamlining the model training process:

1.  **`torch.nn.Module`**: For defining model architectures.
2.  **`torch.nn.Loss`**: For defining loss functions (e.g., `nn.MSELoss`).
3.  **`torch.optim`**: For defining optimization algorithms (e.g., `SGD`, `Adam`).

This notebook builds upon the concepts from previous tutorials, particularly `Gradient Descent with Autograd and Backpropagation.ipynb`, showing how PyTorch's higher-level abstractions simplify the training loop.

## Key Concepts Covered:

1.  **Standard Training Pipeline Steps:** Outlines the typical workflow for training a model in PyTorch.
2.  **Using `nn.MSELoss`:** Demonstrates how to use PyTorch's built-in Mean Squared Error loss function instead of calculating it manually.
3.  **Using `torch.optim`:**
    - Shows how to use optimizers like `torch.optim.SGD` and `torch.optim.Adam`.
    - Explains passing model parameters (`w` or `model.parameters()`) to the optimizer.
    - Illustrates using `optimizer.step()` to update weights based on computed gradients.
    - Highlights the importance of `optimizer.zero_grad()` to clear gradients before the next iteration's backward pass.
4.  **Data Reshaping:** Shows the necessity of reshaping input/output tensors (e.g., from `[N]` to `[N, 1]`) to match the expected input dimensions of `nn.Linear` layers.
5.  **Defining a Model with `nn.Module`:**
    - Introduces `torch.nn.Linear` as a predefined layer for linear transformations (`y = Wx + b`).
    - Shows how to instantiate a simple linear model.
6.  **Integrated Training Loop:** Combines the model definition (`nn.Linear`), loss function (`nn.MSELoss`), and optimizer (`torch.optim.Adam`) into a concise and standard PyTorch training loop.
7.  **Model Parameters:** Accessing and printing model weights and biases using `model.parameters()`.
8.  **Inference:** Making predictions using the trained model (`model(x_test)`).

## How to Use:

1.  Ensure you have PyTorch installed (`pip install torch`).
2.  Open `Training Pipeline - Model, Loss, and Optimizer.ipynb` in a Jupyter environment (like Jupyter Lab or VS Code with the Python extension).
3.  Run the cells sequentially. Pay attention to:
    - The initial example using a single weight tensor `w` with `nn.MSELoss` and `torch.optim.SGD`.
    - The transition to using `nn.Linear` for the model definition.
    - The final training loop using `nn.Linear`, `nn.MSELoss`, and `torch.optim.Adam`.

This notebook bridges the gap between manual gradient descent implementations and the standard, more efficient way of building and training models using PyTorch's core `nn` and `optim` modules.
