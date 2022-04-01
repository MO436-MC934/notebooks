Module 5: The JAX Library
================================================================================

JAX (Just After Execution) is a Python library developed by Google that allows
offloading of numerical code to accelerators using an API very similar to Numpy.
On top of that, users can JIT compile their code, compute the gradient of
functions automatically, and more.

Planning
--------------------------------------------------------------------------------

This module was divided into four lectures:

- **Lecture 1**: We will discuss the basics of JAX and how to use the function
  transformations;
- **Lecture 2**: We will talk about the XLA compiler and the FLAX library.
- **Lecture 3**: We will talk about how JAX works and how to extend it with new
  operators.
- **Lecture 4**: We will walk through an example of implementing LeNet with
  JAX/FLAX.

Installation
--------------------------------------------------------------------------------

We will use Python virtual environments to install JAX and its dependencies.

```bash
# Install virtualenv tool
python3 -m pip install --user virtualenv

# Create a new virtualenv in the current directory
python3 -m virtualenv venv

# Activate the virtualenv
source venv/bin/activate
```

This library is compatible with either CPUs, GPUs, or TPUs as accelerators,
therefore you must install the dependencies for the specific backend you wish to
make use of. If you have a CUDA-compatible GPU, run the following command:

```bash
pip install -r requirements/gpu.txt
```

> **Note**: The instructors were not able to test if JAX works on AMD graphics
> cards. In case it does not work, feel free to use the CPU backend.

However, for x86 CPUs please install the dependencies as follows:

```bash
pip install -r requirements/cpu.txt
```

> **Note**: JAX may not work properly on ARM processors. In case you have no
> other machine to work with, you may resort to [Google Colab][colab].

Running
--------------------------------------------------------------------------------

With your virtual environment activated, simply run:

```bash
jupyter notebook
```

To start the Jupyter server, and your browser will open-up automatically.

[colab]: https://colab.research.google.com/
