# Latest results with modified source code:

Using dt=1
Spatial grid shape: (400, 400, 2)
Features shape: (470, 400, 400, 1)

 Iteration ... |y - Xw|^2 ...  a * |w|_2 ...      |w|_0 ... Total error: |y - Xw|^2 + a * |w|_2
shape of x during ridge regression: (75200000, 17)
         0 ... 1.1238e+12 ... 1.7562e-03 ...          8 ... 1.1238e+12
shape of x during ridge regression: (75200000, 8)
         1 ... 1.1238e+12 ... 1.7562e-03 ...          8 ... 1.1238e+12
### (u)' = 0.374 u + 0.116 uu + -0.042 uuu + 0.080 u_22 + 0.055 u_2222 + 0.084 u_11 + 0.039 u_1122 + 0.046 u_1111

Using dt=1
Spatial grid shape: (448, 448, 2)
Features shape: (470, 448, 448, 1)

 Iteration ... |y - Xw|^2 ...  a * |w|_2 ...      |w|_0 ... Total error: |y - Xw|^2 + a * |w|_2
shape of x during ridge regression: (94330880, 17)
         0 ... 1.3768e+12 ... 1.6990e-03 ...          8 ... 1.3768e+12
shape of x during ridge regression: (94330880, 8)
         1 ... 1.3768e+12 ... 1.6990e-03 ...          8 ... 1.3768e+12
### (u)' = 0.337 u + 0.155 uu + -0.096 uuu + 0.087 u_22 + 0.064 u_2222 + 0.087 u_11 + 0.037 u_1122 + 0.052 u_1111

## Now we can actually see something which includes:
1. The u, u^2, u^3 terms correctly
2. The second degree derivatives (u_22, u_22)
3. The fourth degree derivatives (u_2222, u_1111, u_1122)