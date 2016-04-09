# TensorFlowBuilder

This package assists in writing Julia wrapper code for the `TensorFlow.jl` package.

It uses PyCall.jl, Python's `inspect` module, and the original TensorFlow docstrings to detect Python objects.  It then emits the Julia source code that forms the basis for the package.

The `TensorFlow.jl` source also has further edits and additions.  (This auxiliary package isn't necessary or used at runtime.)


## Installation

This package isn't intended added to `Julia`'s METADATA since it's only needed occasionally to support the development of `TensorFlow.jl`.

Ensure you have installed `PyCall`, then clone it from Github:

```
Pkg.add("PyCall")
Pkg.clone("https://github.com/benmoran/TensorFlowBuilder.jl")
```

You also need to have the Python `tensorflow` package installed, following the [`standard instructions`](http://www.tensorflow.org/get_started/os_setup.md).  Check you are able to import this via PyCall in Julia:

```
usin PyCall
@pyimport tensorflow
```


## Compatibility

The package is developed against Julia 0.4 and CPython 2.7 on Ubuntu 14.04.  Other platforms may work but have not been tested.
