import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Arrays ---
x = np.array([7, 8, 10, 45])
x_arr = np.reshape(x, (2, 2))
print(x_arr)
print(x_arr[0, 1])  # element at [1,2] in R
print(x_arr[2])     # linear index (row-wise in numpy)
print(x_arr[:, 1])  # all rows of column 2

# --- Array Operations ---
y = -x
y_arr = np.reshape(y, (2, 2))
print(y_arr + x_arr)  # elementwise
print(np.sum(x_arr, axis=1))  # row sums

# --- Matrices ---
factory = np.array([[40, 60], [1, 3]])
six_sevens = np.full((2, 3), 7)
print(factory @ six_sevens)

output = np.array([10, 20])
print(factory @ output)
print(output @ factory)

print(factory.T)  # transpose
print(np.linalg.det(factory))  # determinant
print(np.diag(factory))

# Diagonal assignment
np.fill_diagonal(factory, [35, 4])
print(factory)
np.fill_diagonal(factory, [40, 3])

# Identity matrix
print(np.diag([3, 4]))
print(np.identity(2))

# Matrix inversion
inv_factory = np.linalg.inv(factory)
print(inv_factory)
print(factory @ inv_factory)

# Solving Ax = b
available = np.array([1600, 70])
solution = np.linalg.solve(factory, available)
print(solution)
print(factory @ solution)

# Row/Column names using pandas
df_factory = pd.DataFrame(factory, index=["labor", "steel"], columns=["cars", "trucks"])
available_named = pd.Series([1600, 70], index=["labor", "steel"])
output_named = pd.Series([20, 10], index=["trucks", "cars"])
output_aligned = output_named[df_factory.columns]
print(df_factory @ output_aligned)
print((df_factory @ output_aligned <= available_named).all())

print(df_factory.mean(axis=0))
print(df_factory.mean(axis=1))

# --- Lists in Python ---
my_distribution = ["exponential", 7, False]
print(isinstance(my_distribution[0], str))
print(my_distribution[1] ** 2)

# Use dictionary for named list
my_distribution_named = {"family": "exponential", "mean": 7, "is_symmetric": False}
print(my_distribution_named["family"])
my_distribution_named["was_estimated"] = False
my_distribution_named["last_updated"] = "2011-08-30"
del my_distribution_named["was_estimated"]

# --- DataFrames ---
a_matrix = np.array([[35, 10], [8, 4]])
a_df = pd.DataFrame(a_matrix, columns=["v1", "v2"])
a_df["logicals"] = [True, False]
print(a_df["v1"])
print(a_df.iloc[0])
print(a_df.mean())

# Add row
new_row = pd.DataFrame([[-3, -5, True]], columns=a_df.columns)
a_df_extended = pd.concat([a_df, new_row], ignore_index=True)
print(a_df_extended)

# Structures of structures
plan = {"factory": df_factory, "available": available_named, "output": output_named}

# Eigen decomposition
eigen_result = np.linalg.eig(factory)
print("Eigenvalues:", eigen_result[0])
print("Eigenvectors:", eigen_result[1])

# DataFrame Example with US states
from sklearn.datasets import fetch_openml
states = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/states.csv") \
    if "states.csv" in pd.read_csv else pd.DataFrame()  # Placeholder

# Note: in actual R example, state.x77 and related data used
# In practice: you can use seaborn-data 'us_state_abbrev' + economic indicators dataset or similar

# DataFrame access
# Example: states.loc["Wisconsin", "Illiteracy"]
# Conditional selection:
# states[states["region"] == "South"]["Illiteracy"]

# Modify column
# states["HS.Grad"] = states["HS.Grad"] / 100

# With-like context (Pythonic equivalent)
# illit_pct = 100 * (states["HS.Grad"] / (100 - states["Illiteracy"]))

# Plot example
# plt.scatter(states["Frost"], states["Illiteracy"])
# plt.xlabel("Frost")
# plt.ylabel("Illiteracy")
# plt.show()