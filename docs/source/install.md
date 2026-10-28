# Installation

## The standard install

```bash
pip install pyhgf
```

This is the default CPU install and the one most users want. It runs everything in the library: filtering with the {py:class}`pyhgf.model.Network` class, MCMC over network parameters, and the deep predictive coding networks of {py:class}`pyhgf.model.DeepNetwork`.

The development version comes from the master branch:

```bash
pip install "git+https://github.com/ComputationalPsychiatry/pyhgf.git"
```

## JAX versions

pyhgf works with any jax release from 0.4.26 and with Python 3.12 or later. No configuration is required for any of them.

On CPU, jaxlib 0.4.32 to 0.11.2 split the belief propagation scan of the nodalised {py:class}`pyhgf.model.Network` into many small kernels, which makes gradients through it, and so MCMC over network parameters, slower (about ten times for a two-level binary HGF). `import pyhgf` recognises these releases and adds the XLA flag that prevents it to the `XLA_FLAGS` environment variable; jaxlib 0.12.0 and later need no flag.

XLA reads its flags once, when jax starts its backend at the first operation that creates an array. pyhgf therefore has to be imported before that point. Importing jax first, calling `jax.config.update` or wrapping functions in `jax.jit` does not start the backend, so the usual order of imports at the top of a script or notebook works. pyhgf emits a `RuntimeWarning` when a computation has already run.

For reference, the flags added are:

| jaxlib | Flag added by `import pyhgf` |
|---|---|
| 0.4.32 to 0.6.2 | `--xla_cpu_use_thunk_runtime=false` |
| 0.7.0 to 0.11.2 | `--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=1000000000` |

An option already present in `XLA_FLAGS` is left as it is, and setting the environment variable `PYHGF_KEEP_THUNK_RUNTIME` disables this behaviour.

## Running on a GPU

CUDA wheels exist for Linux only; on other platforms the extras below install the CPU version.

### CUDA 12

```bash
pip install "pyhgf[cuda12]"
```

### CUDA 13

The `cuda13` extra requires jax 0.7.2 or later, the first release that provides it:

```bash
pip install "pyhgf[cuda13]"
```
