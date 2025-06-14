import numpy as np
import pandas as pd

# --- if statements ---
x = -3
res = x if x >= 0 else -x
print(res)

x = 1.5
if x**2 < 1:
    result = x**2
else:
    result = 2*abs(x) - 1 if x >= 0 else -2*abs(x) - 1
print(result)

# --- Combining Booleans ---
print((0 > 0) and (42 % 6 == 169 % 13))  # short-circuiting

# --- Iteration ---
table_of_logs = [np.log(i) for i in range(1, 8)]
print(table_of_logs)

# --- Nested for loop for matrix multiplication ---
a = np.random.rand(2, 3)
b = np.random.rand(3, 2)
if a.shape[1] == b.shape[0]:
    c = np.zeros((a.shape[0], b.shape[1]))
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            for k in range(a.shape[1]):
                c[i, j] += a[i, k] * b[k, j]
else:
    raise ValueError("matrices a and b are non-conformable")
print(c)

# --- while loop ---
x = np.array([16, 25, 36])
while max(x) - 1 > 1e-6:
    x = np.sqrt(x)
print(x)

# --- Vectorized operations ---
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b
print(c)

# --- Vectorized conditions: np.where ---
x = np.array([-1.5, -0.5, 0.5, 1.5])
psi = np.where(np.abs(x) <= 1, x**2, 2*np.abs(x) - 1)
print(psi)

# --- Truth values ---
states = pd.DataFrame({'Murder': [10, 4, 8, 3, 6, 2]})
print(np.mean(states["Murder"] > 7))

# --- switch-like structure ---
type_of_summary = "mean"
if type_of_summary == "mean":
    print(states["Murder"].mean())
elif type_of_summary == "median":
    print(states["Murder"].median())
elif type_of_summary == "histogram":
    import matplotlib.pyplot as plt
    plt.hist(states["Murder"])
    plt.show()
else:
    print("I don't understand")

# --- repeat-like loop with break/continue ---
rescued = False
watched = False
while True:
    if watched:
        continue
    print("Help! I am Dr. Morris Culpepper, trapped in an endless loop!")
    if rescued:
        break

# --- Strings ---
president = "Lincoln"
print(len(president))  # nchar
presidents = ["Fillmore", "Pierce", "Buchanan", "Davis", "Johnson"]
print(presidents[2])
print(presidents[3:])

# --- Substrings ---
phrase = "Christmas Bonus"
print(phrase[7:12])
phrase = phrase[:12] + "g" + phrase[13:]
print(phrase)

# --- Vectorized substrings using list comprehension ---
first_two = [p[:2] for p in presidents]
last_two = [p[-2:] if len(p) >= 2 else "" for p in presidents]
print(first_two)
print(last_two)

# --- str.split and str.join ---
scarborough_fair = "parsley, sage, rosemary, thyme"
print(scarborough_fair.split(", "))
multi_split = [s.split(", ") for s in [scarborough_fair, "Garfunkel, Oates", "Clement, McKenzie"]]
print(multi_split)

# --- Combining strings ---
print(str(7.2))
print([str(x) for x in [7.2, 7.2e12]])
print(str(7.2e5))

print([f"{p} {n}" for p, n in zip(presidents, range(41, 46))])
print([f"{p} ({party} {n})" for p, party, n in zip(presidents, ["R", "D"]*3, range(41, 46))])

print([f"{p} ({n})" for p, n in zip(presidents, range(41, 46))])
print("; ".join([f"{p} ({n})" for p, n in zip(presidents, range(41, 46))]))

# --- Regression formula builder ---
def my_formula(dep, indeps, df):
    rhs = "+".join(df.columns[i] for i in indeps)
    return f"{df.columns[dep]} ~ {rhs}"

df = pd.DataFrame(np.random.rand(5, 10), columns=[f"col{i}" for i in range(10)])
print(my_formula(2, [3, 5, 7], df=df))

# --- Text pattern search with regex ---
import re
text_list = ["dog", "cat", "bird", "bat"]
matches = [s for s in text_list if re.search("b", s)]
print(matches)