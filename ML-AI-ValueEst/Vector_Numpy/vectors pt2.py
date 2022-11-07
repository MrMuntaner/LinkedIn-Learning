import numpy as np

# instead of using for loops to work the arrays, numpy takes advantage of the CPU SIMD-
# to multiply chunks of the array in parallel.

# this process is called vectorizing the code.
sizes = np.array([
    1689.0,
    1984.0,
    1581.0,
    1829.0,
    1580.0,
    1621.0,
    2285.0,
    1745.0,
    1747.0,
    998.0,
    2020.0,
    2517.0,
    1835.0
])

sizes = sizes * 0.3

print(sizes)
