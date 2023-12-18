import io
import sys

import pysindy as ps
from pysindy import PDELibrary
from pysindy.optimizers import STLSQ
from pysindy.differentiation import SpectralDerivative
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

def patch_sindy(path, res):
    PDE_dict = {}
    output_buffer = io.StringIO()
    sys.stdout = output_buffer

    features_dir = path

    frames = []
    for filename in os.listdir(features_dir):
        print(filename)
        #if filename.endswith(".png"):
        if filename.endswith(".jpg"):
            img = Image.open(os.path.join(features_dir, filename))
            frames.append(img)

    features = np.stack(frames, axis=0)
    features = np.reshape(features, (50, res, res, 3))  # Artificially replicating pixel values as SINDy is a bit dumb
    spatial_grid_x, spatial_grid_y = np.meshgrid(np.arange(res), np.arange(res))
    spatial_grid = np.stack((spatial_grid_x, spatial_grid_y), axis=-1)

    # Define feature library
    lib = ps.PDELibrary(
        library_functions=[
            lambda x: x,
            lambda x: x ** 2,
            lambda x: x ** 3,
        ],
        function_names=[
            lambda x: x,
            lambda x: x + x,
            lambda x: x + x + x,
        ],
        derivative_order=4,
        spatial_grid=spatial_grid,
        include_interaction=False,
        differentiation_method=ps.SpectralDerivative,
    )

    # Define optimizer, remember alpha is the regularization parameter, threshold is the sparsity parameter.
    opt = STLSQ(threshold=0.01, alpha=0.01, verbose=True)

    # Fitting the model
    x = np.asarray(features)
    dt = np.arange(0, x.shape[-2] * 0.05, 0.05)
    model = ps.SINDy(feature_library=lib, optimizer=opt, feature_names=["u"])
    x_dot = model.differentiate(x=x)
    model.fit(x=x, x_dot=x_dot, t=dt)
    print("HI")
    model.print()

    output_content = output_buffer.getvalue()
    output_content = output_content.split("Total error: |y - Xw|^2 + a * |w|_2\n")[-1]
    sys.stdout = sys.__stdout__
    key = str(res)
    PDE_dict[key] = output_content
    return PDE_dict