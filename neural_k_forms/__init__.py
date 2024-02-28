"""General definitions for the package."""

import os.path

# Root directory for data sets. `pytorch-geometric` has the capability
# to automatically save them in a cache dir. This construction ensures
# that all data sets are placed in a data folder in the repo root dir.
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")

__all__ = ["generate_cochain_data_matrix"]
