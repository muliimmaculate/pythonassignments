# --- Basic Arithmetic & Operators ---
print(7 + 5)       # 12
print(7 - 5)       # 2
print(7 * 5)       # 35
print(7 ** 5)      # 16807
print(7)       # 1.4
print(7 % 5)       # 2 (modulo)
print(7 // 5)      # 1 (integer division)

# --- Comparison Operators ---
print(7 > 5)       # True
print(7 < 5)       # False
print(7 >= 7)      # True
print(7 <= 5)      # False
print(7 == 5)      # False
print(7 != 5)      # True

# --- Boolean Operators ---
print((5 > 7) and (6 * 7 == 42))  # False
print((5 > 7) or (6 * 7 == 42))   # True

# --- Type Checking and Casting ---
print(type(7))                         # <class 'int'>
print(isinstance(7, (int, float)))    # True
import math
print(math.isnan(7))                  # False
print(math.isnan(float('nan')))      # True

print(isinstance(7, str))             # False
print(isinstance("7", str))          # True
print(isinstance("seven", str))      # True

num_str = str(5 / 6)
print(num_str)                        # "0.8333333333333334"
print(float(num_str))                # 0.8333333333333334
print(6 * float(num_str))            # 5.0
print(5/6 == float(num_str))         # False

# --- Variables and Assignment ---
pi = 3.141593
print(pi)                             # 3.141593
print(pi * 10)                        # 31.41593
print(math.cos(pi))                  # -1.0

approx_pi = 22 / 7
diameter_in_cubits = 10
print(approx_pi * diameter_in_cubits)  # 31.42857

circumference_in_cubits = approx_pi * diameter_in_cubits
print(circumference_in_cubits)         # 31.42857

circumference_in_cubits = 30
print(circumference_in_cubits)         # 30

# --- Workspace (Variable List) ---
print(dir())                          # List of current variables

# --- Vectors with NumPy ---
import numpy as np

x = np.array([7, 8, 10, 45])
print(x[0])                           # 7
print(np.delete(x, 3))               # Removes 4th element

weekly_hours = np.zeros(5)
weekly_hours[4] = 8
print(weekly_hours)

# Vector arithmetic
y = np.array([-7, -8, -10, -45])
print(x + y)                          # [0, 0, 0, 0]
print(x * y)                          # [-49, -64, -100, -2025]

# Recycling (broadcasting)
print(x + np.tile(np.array([-7, -8]), 2))   # [0, 0, 3, 37]
print(x ** np.array([1, 0, -1, 0.5]))       # [7. 1. 0.1 6.708...]

# Comparisons
print(x > 9)                          # [False False  True  True]
print((x > 9) & (x < 20))             # [False False  True False]
print(x == -y)                        # [ True  True  True  True]
print(np.array_equal(x, -y))         # True
print(np.allclose([0.5-0.3, 0.3-0.1], [0.3-0.1, 0.5-0.3]))  # True

# Functions on Vectors
print(np.mean(x))
print(np.median(x))
print(np.std(x))
print(np.var(x))
print(np.max(x))
print(np.min(x))
print(len(x))
print(np.sum(x))
print(np.sort(x))

# Boolean vector usage
print(x[x > 9])                      
print(y[x > 9])                     
places = np.where(x > 9)[0]
print(places)
print(y[places])

# Named Components with Dictionaries
x_dict = {"v1": 7, "v2": 8, "v3": 10, "fred": 45}
print(x_dict["fred"])               # 45
print([x_dict[key] for key in ["fred", "v1"]])  # [45, 7]
print(sorted(x_dict.keys()))        # ['fred', 'v1', 'v2', 'v3']
print([i for i, k in enumerate(x_dict.keys()) if k == "fred"])  # [3]

# Peculiarities of floating-point numbers
print(0.45 == 3 * 0.15)             # False
print(0.45 - 3 * 0.15)              # ~5.55e-17
print((0.5 - 0.3) == (0.3 - 0.1))   # False
print(math.isclose(0.5 - 0.3, 0.3 - 0.1))  # True

# Integer Check
print(isinstance(7, int))           # True
print(int(7))                       # 7
print(round(7) == 7)                # True
