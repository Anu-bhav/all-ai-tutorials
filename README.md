# all-ai-tutorials

All my AI related tutorials for different libraries like pytorch, scikit learn and others

## Setting up environment

```bash
uv init .
uv venv .venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv add torch numpy scikit-learn pandas matplotlib
uv run check_dependencies.py
```

```bash
$ uv run check_dependencies.py
--- Checking Essential Libraries ---
[ OK ] Imported 'matplotlib' successfully (Version: 3.10.1)
[ OK ] Imported 'numpy' successfully (Version: 2.2.5)
[ OK ] Imported 'pandas' successfully (Version: 2.2.3)
[ OK ] Imported 'sklearn' successfully (Version: 1.6.1)
[ OK ] Imported 'torch' successfully (Version: 2.6.0+cu124)
------------------------------------

--- Checking PyTorch CUDA Support ---
PyTorch Version: 2.6.0+cu124
CUDA Available: True
CUDA Device Count: 1
CUDA Device Name: NVIDIA GeForce RTX 4070 Ti
------------------------------------

--- Summary ---
All essential libraries imported successfully!
Environment looks good to go for the tutorial.
```
