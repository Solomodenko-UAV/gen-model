import os 


on_cpu = os.environ.get("USE_GPU") != False and os.environ.get("USE_GPU") != 'True'

if on_cpu:
    import numpy as cp
else:
    import cupy as cp
    

class FPN:
    def __init__(self):
        