# cochain_representation_learning

## Installation with `pip` or `poetry`

With `poetry install` or `pip install .`, the base packages will be set up.
However, we need to install `torch` and `torch-geometric` *manually*, since
their installation procedure differs based on CPU or GPU versions.

For the CPU versions, this should work:

```
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
$ pip install torch_geometric
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
$ pip install torchmetrics pytorch-lightning
```
