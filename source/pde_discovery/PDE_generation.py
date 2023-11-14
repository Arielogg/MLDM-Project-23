import pysindy as ps
from pysindy import PDELibrary
from pysindy.optimizers import STLSQ
from pysindy.differentiation import SpectralDerivative
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

#Change this to your own directories, frame imports
features_dir = r"../../generations/cross_generations"  #Cross et al. simulator, epsilon 0.3 gamma 0.2
#features_dir = r'brandao_generations' #Brandao et al. simulator, epsilon 0.3 gamma 0.2

#Only available for the brandao generator, timesteps
transient_time_file = "../../generations/brandao_generations/transient_times.csv"
t = np.loadtxt(transient_time_file, delimiter=",")
dt = np.mean(np.diff(t))
dt = round(dt, 1)

res = 448
frames = []

for filename in os.listdir(features_dir):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(features_dir, filename))
        #width, height = img.size
        #img = img.resize((res, res)).convert('L') # Just keeping this line in case we want to change image sizes
        frames.append(img)

#Creating spatial grid
features = np.stack(frames, axis=0)
features = np.reshape(features, (470, res, res, 1)) # Artificially replicating pixel values as SINDy is a bit dumb
spatial_grid_x, spatial_grid_y = np.meshgrid(np.arange(res), np.arange(res))
spatial_grid = np.stack((spatial_grid_x, spatial_grid_y), axis=-1)
print("Spatial grid shape:", np.shape(spatial_grid))
print("Features shape:", np.shape(features))

#Plotting the last frame if you want to see what it looks like
'''
plt.imshow(features[0,:,:,0], cmap='gray')
plt.colorbar(label='Intensity')
plt.show()'''

#Define feature library
lib = ps.PDELibrary(
    library_functions=[
        lambda x: x,
        lambda x: x ** 2,
        lambda x: x ** 3,
    ],
    function_names = [
        lambda x: x,
        lambda x: x + x,
        lambda x: x + x + x,
    ],
    derivative_order=4,
    spatial_grid=spatial_grid,
    include_interaction=False,
    differentiation_method=ps.SpectralDerivative,
)

#Define optimizer, remember alpha is the regularization parameter, threshold is the sparsity parameter.
opt = STLSQ(threshold=0.01, alpha=0.01, verbose=True)

#Fitting the model
print("Fitting model...")
x = np.asarray(features)
dt = np.arange(0, x.shape[-2]*0.05, 0.05)

model = ps.SINDy(feature_library=lib, optimizer=opt, feature_names=["u"])

x_dot = model.differentiate(x=x)  ## check pysindy.pi function differentiate for adjustment of timestep, not automatic.
plt.imshow(x_dot[1,:,:,0], cmap='gray')
plt.colorbar(label='Intensity')
plt.show()

print("Shape of x_dot:", np.shape(x_dot))
model.fit(x=x, x_dot=x_dot, t=dt) ## Check spectral_derivative.py for the adjustment of differentiation axis and timestep, also not automatic
model.print()

## FIXED DIFFERENTIATION. Now must fix STSLQ to work with 3D arrays to calculate the 'total error' because as of right now it's just comparing pixels
# it's just flattening anything as a mega long thing