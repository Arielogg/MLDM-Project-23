import pysindy as ps
from pysindy import PDELibrary, WeakPDELibrary
from pysindy.optimizers import STLSQ
from pysindy.differentiation import SpectralDerivative
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

#Change this to your own directories, frame imports
trajectory_1 = r"../../generations/cross_generations/e03g02"  #Cross et al. simulator, epsilon 0.3 gamma 0.2
trajectory_2 = r"../../generations/cross_generations/e011_g_061"  #Cross et al. simulator, epsilon 0.11 gamma -0.61
trajectory_3 = r"../../generations/cross_generations/e014_g0"  #Cross et al. simulator, epsilon 0.14 gamma 0
trajectory_4 = r"../../generations/cross_generations/e015_g033"  #Cross et al. simulator, epsilon 0.15 gamma 0.33

""" Features directory of the Raw Images """
features_dir = r"C:\Users\ariel\PycharmProjects\MLDM_Project\data\multi_trajectories\20_2\cut_1"

res = 224

#List of trajectory directories
trajectory_dirs = [trajectory_1, trajectory_2, trajectory_3, trajectory_4]

#Initialize an empty list to store the frames
frames = []

#Working with multiple trajectories
'''for trajectory_dir in trajectory_dirs:
    print("Importing frames from trajectory directory:", trajectory_dir)
    filenames = sorted(os.listdir(trajectory_dir))
    for filename in filenames[0:200]:
        if filename.endswith(".png"):
            img = Image.open(os.path.join(trajectory_dir, filename))
            width, height = img.size
            img = img.resize((res, res)).convert('L')
            img = np.asarray(img)
            img = img / 255
            frames.append(img)
features = np.stack(frames, axis=0)
features = np.reshape(features, (len(features)//4, res, res, 4))
print("Features shape:", np.shape(features))
'''
# Working with a single trajectory
filenames = sorted(os.listdir(features_dir))
for filename in filenames:
    if filename.endswith(".jpg"):
        img = Image.open(os.path.join(features_dir, filename))
        width, height = img.size
        img = img.resize((res, res)).convert('L') # Just keeping this line in case we want to change image sizes
        img = np.asarray(img)
        img = img / 255
        frames.append(img)

features = np.stack(frames[0:23], axis=0)
features = np.reshape(features, (len(features), res, res, 1))
print("Features shape:", np.shape(features))

#Creating spatial and temporal grids for the derivatives
spatial_grid_x, spatial_grid_y = np.meshgrid(np.arange(res), np.arange(res))
spatial_grid = np.stack((spatial_grid_x, spatial_grid_y), axis=-1)
dt = np.arange(0, features.shape[-4]*0.5, 0.5)
print("Spatial grid shape:", np.shape(spatial_grid))
print("Temporal grid shape: ", np.shape(dt))

#Plotting the first frame if you want to see what it looks like
'''plt.imshow(features[0,:,:], cmap='gray')
plt.colorbar(label='Intensity')
plt.show()'''

#Define feature library
feature_lib = ps.PDELibrary(
    library_functions=[lambda x: x, lambda x: x ** 2, lambda x: x ** 3],
    derivative_order=4,
    spatial_grid=spatial_grid,
    include_interaction=False,
    function_names=[lambda x: x,lambda x: x + x,lambda x: x + x + x],
    multiindices=[[0,2], [2,0], [2,2], [4,0], [0,4]],
    differentiation_method=SpectralDerivative,
)

parameter_lib = ps.PDELibrary(
    library_functions=[lambda x: x],
    include_interaction=False,
    function_names=[lambda x: x],
    include_bias=True,
)

lib = ps.ParameterizedLibrary(
    parameter_library=parameter_lib,
    feature_library=feature_lib,
    num_parameters=2,
    num_features=1,
)

#Define optimizer, remember alpha is the regularization parameter, threshold is the sparsity parameter.
print()
print("Fitting model...")
opt = ps.STLSQ(threshold=0.001, alpha=0.01, normalize_columns=False, verbose=True)
model = ps.SINDy(feature_library=lib, optimizer=opt, feature_names=["u", "F", "t"]) # Try a bunch of alpha/threshold combinations

x = np.asarray(features)
x_dot = model.differentiate(x=x)  ## check pysindy.pi function differentiate for adjustment of timestep, not automatic.
#u = [[0.3, 0.2],[0.11, -0.61],[0.14, 0],[0.15,0.33]]
#u = [0.20, 0.02]
## Remember to change the axis to axis+1 in the spectral derivative function
print("Shape of x_dot", np.shape(x_dot))

'''plt.imshow(x_dot[10,:,:,0], cmap='gray')
plt.colorbar(label='Intensity')
plt.show()
'''
model.fit(x, x_dot=x_dot, u=u, multiple_trajectories=False)  # Check spectral_derivative.py for the adjustment of differentiation axis and timestep, also not automatic
model.print()
print()

weights = model.coefficients()
normalized_weights = weights / np.linalg.norm(weights)
print(weights)
print(normalized_weights)

model.coefficients = normalized_weights
model.print()

## Get simulation and plot
#sim = model.simulate(x[0,:,:,0], t=dt, u=u)
#print("Shape of simulation:", np.shape(sim))
#plt.imshow(sim[0,:,:,0], cmap='gray')
#plt.colorbar(label='Intensity')
#plt.show()

#### HYPERPARAMETER SEARCH ####

'''# List of threshold-alpha combinations
thresholds = [0.000001, 0.0001, 0.001, 0.01, 0.1, 1]
alphas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]

# Iterate over every threshold-alpha combination
for threshold in thresholds:
    for alpha in alphas:
        print("Threshold:", threshold)
        print("Alpha:", alpha)
        opt = ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=False, verbose=True)
        model = ps.SINDy(feature_library=lib, optimizer=opt, feature_names=["u", "F", "t"])
        model.fit(x, x_dot=x_dot, u=u, multiple_trajectories=False)
        model.print()
        print()
'''