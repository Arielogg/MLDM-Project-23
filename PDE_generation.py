import pysindy as ps
from pysindy import PDELibrary
from pysindy.optimizers import STLSQ
from pysindy.differentiation import SpectralDerivative
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

#Change this to your own directories, frame imports
#features_dir = r"cross_generations" #Cross et al. simulator, epsilon 0.3 gamma 0.2
features_dir = r'brandao_generations' #Brandao et al. simulator, epsilon 0.3 gamma 0.2

#Only available for the brandao generator, timesteps
transient_time_file = "brandao_generations/transient_times.csv"
t = np.loadtxt(transient_time_file, delimiter=",")
dt = np.mean(np.diff(t))
dt = round(dt, 1)
print("Average dt:", dt/1000, "s")

features = []

#Most of this preprocessing loop shouldn't be necessary for images that are already grayscale and cropped to a 'desirable' size.
for filename in os.listdir(features_dir):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(features_dir, filename))
        width, height = img.size
        #img = img.crop((width/4, height/4, 3*width/4, 3*height/4))  #For the cross generator, must crop the images in half for reasons (don't know which)
        img = img.crop((width/8, height/8, 7*width/8, 7*height/8))   #For the brandao generator, must crop the images by like 1/4 for reasons (don't know which)
        img = img.resize((128, 128)).convert('L')  #Resizing and converting to grayscale
        img = np.array(img) / 255
        features.append(np.array(img))

features = np.array(features[0:50]) #Chosing the first 50 frames

#Plotting the last frame if you want to see what it looks like
'''
plt.imshow(features[-1], cmap='gray')
plt.colorbar(label='Intensity')
plt.show()
features = np.transpose(features, (1, 2, 0))'''

print("Features shape:", features.shape)

#Define feature library
lib = ps.PDELibrary(
    library_functions=[
        lambda u: u,
        lambda u: u**2,
        lambda u: u**3
    ],
    derivative_order=4,
    spatial_grid=features,
    include_interaction=False,
    function_names=[
        lambda u: u,
        lambda u: u + u,
        lambda u: u + u + u
    ],
    differentiation_method=ps.SpectralDerivative,
)

#Define optimizer, remember alpha is the regularization parameter, threshold is the sparsity parameter.
opt = STLSQ(threshold=0.1, alpha=0.001, verbose=True)

#Fitting the model
print("Fitting model...")
model = ps.SINDy(feature_library=lib, optimizer=opt)
model.fit(x=features, t=dt) #t is the size of the timestep
model.print()