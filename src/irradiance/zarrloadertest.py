import zarr
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import pandas as pd

# OBS: I was supposed to create a new conda environment, but it was throwing an error and I could not fix it in a short time. When running
# a cell, it will ask to select an interpreter --> use the base one with Python 3.10.11 (or create a custom env with a python version such
# as Will did)

# access data (currently mounted on the VM) --> no need to call GCP bucket storage for now
store = zarr.DirectoryStore('/mnt/sdomlv2_small/sdomlv2_small.zarr')
root = zarr.group(store)
print(root.tree())

# How to extract "keys" in the tree. OBS: the for loop is necessary to get the name. Calling for the keys (i.e., just root.keys()) does not provide the 
# name itself (in this case, "2010") like it would for a dictionary object
tree_keys = root.keys()

for name in tree_keys:
  print(type(name))
  print(name)