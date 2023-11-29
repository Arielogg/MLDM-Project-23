import pysindy as ps
from pysindy import PDELibrary
from pysindy.optimizers import STLSQ
from pysindy.differentiation import SpectralDerivative
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import sys
import io

# Change this to your own directories, frame imports
""" Features directory of the Test-Images """
# features_dir = r"../../generations/cross_generations"  #Cross et al. simulator, epsilon 0.3 gamma 0.2
# features_dir = r'brandao_generations' #Brandao et al. simulator, epsilon 0.3 gamma 0.2

""" Features directory of the generated changemaps """
# features_dir = r"../../generations/changemap_generations/cropped_cropped/result_24_cut_4_png"
# features_dir = r"../../generations/changemap_generations/cropped_resized/result_24_cut_4_png"

""" Features directory of the generated registered by deformation (This images are .jpg)"""
# features_dir = r"../../generations/deformation_generations/registered/20"
features_dir = r"../../generations/deformation_generations/deformations/24"

# Only available for the brandao generator, timesteps

# transient_time_file = "../../generations/brandao_generations/transient_times.csv"
# t = np.loadtxt(transient_time_file, delimiter=",")
# dt = np.mean(np.diff(t))
# dt = round(dt, 1)

""" Resolution for Test-Images """
# res = 448

""" Resolution for Changemaps """
res = 224

frames = []

for filename in os.listdir(features_dir):
    #if filename.endswith(".png"):
    if filename.endswith(".jpg"):
        img = Image.open(os.path.join(features_dir, filename))
        # width, height = img.size
        # img = img.resize((res, res)).convert('L') # Just keeping this line in case we want to change image sizes
        frames.append(img)

# Creating spatial grid

features = np.stack(frames, axis=0)
# features = np.reshape(features, (470, res, res, 1)) # Artificially replicating pixel values as SINDy is a bit dumb
features = np.reshape(features, (49, res, res, 1))  # Artificially replicating pixel values as SINDy is a bit dumb
spatial_grid_x, spatial_grid_y = np.meshgrid(np.arange(res), np.arange(res))
spatial_grid = np.stack((spatial_grid_x, spatial_grid_y), axis=-1)
print("Spatial grid shape:", np.shape(spatial_grid))
print("Features shape:", np.shape(features))

# Plotting the last frame if you want to see what it looks like

"""
plt.imshow(features[0,:,:,0], cmap='gray')
plt.colorbar(label='Intensity')
plt.show()
"""

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
# threshold = 0.01, alpha = 0.01
thresholds = [0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11,
              0.12, 0.13, 0.14, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
alphas = [0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11,
              0.12, 0.13, 0.14, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 5, 10, 50, 100, 1000, 10000]
max_iters = [1, 2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 70, 100, 200, 500, 700, 1000, 5000, 10000]
normalize_columns = [True, False]
PDE_dict = {}

""" Threshold """
complexity_threshold = []
error_threshold = []
for threshold in thresholds:
    output_buffer = io.StringIO()
    sys.stdout = output_buffer

    # Define optimizer, remember alpha is the regularization parameter, threshold is the sparsity parameter.
    opt = STLSQ(threshold=threshold, alpha=0.01, verbose=True)

    # Fitting the model
    print("Fitting model...")
    x = np.asarray(features)
    dt = np.arange(0, x.shape[-2] * 0.05, 0.05)

    model = ps.SINDy(feature_library=lib, optimizer=opt, feature_names=["u"])

    x_dot = model.differentiate(x=x)

    print("Shape of x_dot:", np.shape(x_dot))
    model.fit(x=x, x_dot=x_dot, t=dt)
    model.print()

    output_content = output_buffer.getvalue()
    output_content = output_content.split("Total error: |y - Xw|^2 + a * |w|_2\n")[-1]
    sys.stdout = sys.__stdout__

    key = str("threshold: " + str(threshold) + ", alpha: " + str(0.01))
    PDE_dict[key] = output_content

    output_content_pde = output_content.split("(u)'")[-1]
    output_content_error = output_content.split("(u)'")[0]
    output_content_error = output_content_error.split("...")[-1]

    complexity_threshold.append(output_content_pde.count('+'))
    error_threshold.append(output_content_error)

""" Alpha """
complexity_alpha = []
error_alpha = []
for alpha in alphas:
    output_buffer = io.StringIO()
    sys.stdout = output_buffer

    # Define optimizer, remember alpha is the regularization parameter, threshold is the sparsity parameter.
    opt = STLSQ(threshold=0.01, alpha=alpha, verbose=True)

    # Fitting the model
    print("Fitting model...")
    x = np.asarray(features)
    dt = np.arange(0, x.shape[-2] * 0.05, 0.05)

    model = ps.SINDy(feature_library=lib, optimizer=opt, feature_names=["u"])

    x_dot = model.differentiate(x=x)

    print("Shape of x_dot:", np.shape(x_dot))
    model.fit(x=x, x_dot=x_dot, t=dt)
    model.print()

    output_content = output_buffer.getvalue()

    output_content = output_content.split("Total error: |y - Xw|^2 + a * |w|_2\n")[-1]

    sys.stdout = sys.__stdout__

    key = str("threshold: " + str(0.01) + ", alpha: " + str(alpha))
    PDE_dict[key] = output_content

    output_content_pde = output_content.split("(u)'")[-1]
    output_content_error = output_content.split("(u)'")[0]
    output_content_error = output_content_error.split("...")[-1]

    complexity_alpha.append(output_content_pde.count('+'))
    error_alpha.append(output_content_error)

""" Max_iter """
complexity_maxIter = []
error_maxIter = []
for max_iter in max_iters:
    output_buffer = io.StringIO()
    sys.stdout = output_buffer

    # Define optimizer, remember alpha is the regularization parameter, threshold is the sparsity parameter.
    opt = STLSQ(threshold=0.01, alpha=0.01, max_iter=max_iter, verbose=True)

    # Fitting the model
    print("Fitting model...")
    x = np.asarray(features)
    dt = np.arange(0, x.shape[-2] * 0.05, 0.05)

    model = ps.SINDy(feature_library=lib, optimizer=opt, feature_names=["u"])

    x_dot = model.differentiate(x=x)

    print("Shape of x_dot:", np.shape(x_dot))
    model.fit(x=x, x_dot=x_dot, t=dt)
    model.print()

    output_content = output_buffer.getvalue()

    output_content = output_content.split("Total error: |y - Xw|^2 + a * |w|_2\n")[-1]

    sys.stdout = sys.__stdout__

    key = str("threshold: " + str(0.01) + ", alpha: " + str(0.01) + ", max_iter: " + str(max_iter))
    PDE_dict[key] = output_content

    output_content_pde = output_content.split("(u)'")[-1]
    output_content_error = output_content.split("(u)'")[0]
    output_content_error = output_content_error.split("...")[-1]

    complexity_maxIter.append(output_content_pde.count('+'))
    error_maxIter.append(output_content_error)

""" Normalize Columns """
complexity_normalizeCols = []
error_normalizeCols = []
for normalize_column in normalize_columns:
    output_buffer = io.StringIO()
    sys.stdout = output_buffer

    # Define optimizer, remember alpha is the regularization parameter, threshold is the sparsity parameter.
    opt = STLSQ(threshold=0.01, alpha=0.01, normalize_columns=normalize_column, verbose=True)

    # Fitting the model
    print("Fitting model...")
    x = np.asarray(features)
    dt = np.arange(0, x.shape[-2] * 0.05, 0.05)

    model = ps.SINDy(feature_library=lib, optimizer=opt, feature_names=["u"])

    x_dot = model.differentiate(x=x)

    print("Shape of x_dot:", np.shape(x_dot))
    model.fit(x=x, x_dot=x_dot, t=dt)
    model.print()

    output_content = output_buffer.getvalue()

    output_content = output_content.split("Total error: |y - Xw|^2 + a * |w|_2\n")[-1]

    sys.stdout = sys.__stdout__

    key = str("threshold: " + str(0.01) + ", alpha: " + str(0.01) + ", normalize_Cols: " + str(normalize_column))
    PDE_dict[key] = output_content

    output_content_pde = output_content.split("(u)'")[-1]
    output_content_error = output_content.split("(u)'")[0]
    output_content_error = output_content_error.split("...")[-1]

    complexity_normalizeCols.append(output_content_pde.count('+'))
    error_normalizeCols.append(output_content_error)

fout = "Optimizer_effect_on_PDE.txt"
fo = open(fout, "w")
for k, v in PDE_dict.items():
    fo.write(str(k) + ': \n ' + str(v) + '\n\n')
fo.close()


plt.plot(thresholds, complexity_threshold)
plt.title("Threshold vs. Complexity")
plt.xlabel("Threshold")
plt.ylabel("Complexity")
plt.semilogx()
plt.show()

plt.plot(thresholds, error_threshold)
plt.title("Threshold vs. Errors")
plt.xlabel("Threshold")
plt.ylabel("Error")
plt.semilogx()
plt.show()

plt.plot(alphas, complexity_alpha)
plt.title("Alpha vs. Complexity")
plt.xlabel("Alpha")
plt.ylabel("Complexity")
plt.semilogx()
plt.show()

plt.plot(alphas, error_alpha)
plt.title("Alpha vs. Errors")
plt.xlabel("Alpha")
plt.ylabel("Error")
plt.semilogx()
plt.show()

plt.plot(max_iters, complexity_maxIter)
plt.title("Max_iter vs. Complexity")
plt.xlabel("Max_iter")
plt.ylabel("Complexity")
plt.semilogx()
plt.show()

plt.plot(max_iters, error_maxIter)
plt.title("Max_iter vs. Errors")
plt.xlabel("Max_iter")
plt.ylabel("Error")
plt.semilogx()
plt.show()

plt.plot(normalize_columns, complexity_normalizeCols)
plt.title("Normalize_cols vs. Complexity")
plt.xlabel("Normalize_cols")
plt.ylabel("Complexity")
plt.semilogx()
plt.show()

plt.plot(normalize_columns, error_normalizeCols)
plt.title("Normalize_cols vs. Errors")
plt.xlabel("Normalize_cols")
plt.ylabel("Error")
plt.semilogx()
plt.show()
