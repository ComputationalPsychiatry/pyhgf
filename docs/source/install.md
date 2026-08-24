# Installation

```{eval-rst}
:html_theme.sidebar_secondary.remove:
```

## The standard install

```bash
pip install pyhgf
```

This is the default CPU install and the one most users want. It runs everything in the library: filtering with the {py:class}`pyhgf.model.Network` class, MCMC over network parameters, and the deep predictive coding networks of {py:class}`pyhgf.model.DeepNetwork`.

The development version comes from the master branch:

```bash
pip install "git+https://github.com/ComputationalPsychiatry/pyhgf.git"
```

## Running on a GPU

CUDA wheels exist for Linux only. GPU work means the deep network path, which use the vectorised implementation that is not affected by the limits discussed above, so these installs are free to use any jax release.

### CUDA 12

The `cuda12` extra resolves inside the cap and needs nothing special:

```bash
pip install "pyhgf[cuda12]"
```

### CUDA 13

JAX only ships its `cuda13` extra from version `0.7.2`, which will require an override on the command lin with UV:

```bash
uv pip install "pyhgf[cuda13]" --override <(printf 'jax[cuda13]\njaxlib\n')
```

With pip, which has no override mechanism, install pyhgf first and upgrade jax over the top:

```bash
pip install pyhgf
pip install --upgrade "jax[cuda13]"
```
