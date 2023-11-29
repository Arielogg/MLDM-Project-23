import pysindy as ps
from pysindy import PDELibrary
from pysindy.optimizers import STLSQ
from pysindy.differentiation import SpectralDerivative
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import io
import sys

from pysindy.utils import lorenz
from scipy.integrate import solve_ivp

""" !!!!!!!!! ATTENTION !!!!!!!!!
To use multiple_trajectories, you have to replace line 673 in pysindy.py (External Libraries/pysindy/pysindy.py) by the following:
def differentiate(self, x, t=None, multiple_trajectories=True):
"""

# Change this to your own directories, frame imports
""" Features directory of the Test-Images """
# features_dir = r"../../generations/cross_generations"  #Cross et al. simulator, epsilon 0.3 gamma 0.2
# features_dir = r'brandao_generations' #Brandao et al. simulator, epsilon 0.3 gamma 0.2

""" Features directory of the generated changemaps """
features_dir_1 = r"../../generations/changemap_generations/cropped_cropped/result_24_cut_1_png"
features_dir_2 = r"../../generations/changemap_generations/cropped_cropped/result_24_cut_2_png"
features_dir_3 = r"../../generations/changemap_generations/cropped_cropped/result_24_cut_3_png"
features_dir_4 = r"../../generations/changemap_generations/cropped_cropped/result_24_cut_4_png"

features_dirs = [features_dir_1, features_dir_2, features_dir_3, features_dir_4]

""" Features directory of the generated registered by deformation (This images are .jpg)"""
# features_dir = r"../../generations/deformation_generations/registered/20"
# features_dir = r"../../generations/deformation_generations/deformations/24"

""" Resolution for Changemaps & Deformation """
res = 224

traj_len = 49
#traj_len = 50


def get_features(features_dir):
    frames = []
    for filename in os.listdir(features_dir):
        if filename.endswith(".png"):
        # if filename.endswith(".jpg"):
            img_1 = Image.open(os.path.join(features_dir, filename))
            frames.append(img_1)

    features = np.stack(frames, axis=0)
    # features = np.reshape(features, (470, res, res, 1)) # Artificially replicating pixel values as SINDy is a bit dumb
    features = np.reshape(features, (traj_len, res, res, 1))  # Artificially replicating pixel values as SINDy is a bit dumb
    return np.asarray(features)


# Creating spatial grid
spatial_grid_x, spatial_grid_y = np.meshgrid(np.arange(res), np.arange(res))
spatial_grid = np.stack((spatial_grid_x, spatial_grid_y), axis=-1)

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
x1 = get_features(features_dir_1)
x2 = get_features(features_dir_2)
x3 = get_features(features_dir_3)
x4 = get_features(features_dir_4)

x_train_multi = [x1, x2, x3, x4]

dt = np.arange(0, x_train_multi[0].shape[-2] * 0.05, 0.05)

model = ps.SINDy(feature_library=lib, optimizer=opt, feature_names=["u"])
x_dot = model.differentiate(x=x_train_multi)
model.fit(x=x_train_multi, x_dot=x_dot, t=dt, multiple_trajectories=True)
model.print()
