import matplotlib.pyplot as plt

patch_sizes = []
errors = []

with open('Patch_effect_on_Error.txt') as fp:
    for line in fp:
        patch_size = int(line.split(":")[0])
        patch_sizes.append(patch_size)
        error = float(line.split(":")[-1])
        error_normalized = error/(patch_size*patch_size*48)
        errors.append(error_normalized)

plt.plot(patch_sizes, errors)
plt.title("Patch size vs. Normalized Error")
plt.xlabel("Patch size")
plt.ylabel("Normalized Error")
plt.grid("minor")
plt.show()
