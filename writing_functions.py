import numpy as np

# --- Cube Function ---
def cube(x):
    return np.power(x, 3)

print(cube(3))
print(cube(np.arange(1, 11)))
print(cube(np.array([[1, 3, 5, 7], [2, 4, 6, 8]])))

# --- Robust Loss Function Variants ---
def psi_1(x):
    return np.where(x**2 > 1, 2*np.abs(x) - 1, x**2)

def psi_2(x, c=1):
    return np.where(x**2 > c**2, 2*c*np.abs(x) - c**2, x**2)

def psi_3(x, c=1):
    assert isinstance(c, (int, float)) and c > 0, "c must be a positive number"
    return np.where(x**2 > c**2, 2*c*np.abs(x) - c**2, x**2)

z = np.array([-0.5, -5, 0.9, 9])
print(psi_1(z))
print(psi_2(z, c=1))
print(np.all(psi_2(z, c=1) == psi_2(z)))
print(np.all(psi_2(x=z, c=2) == psi_2(c=2, x=z)))

# --- Function Environments Example ---
x = 7
y = ["A", "C", "G", "T", "U"]

def adder(y):
    return x + y

print(adder(1))  # Uses x from outer scope
print(x)
print(y)

def circle_area(r):
    return np.pi * r**2

print(circle_area(np.array([1, 2, 3])))

# --- Estimating Power Law Exponent ---
def estimate_scaling_exponent(a, y0=6611, response=None, predictor=None,
                               max_iter=100, deriv_step=0.001, step_scale=1e-12, stop_deriv=0.01):
    assert response is not None and predictor is not None, "response and predictor must be provided"

    def mse(a_val):
        return np.mean((response - y0 * predictor ** a_val) ** 2)

    for iteration in range(1, max_iter + 1):
        deriv = (mse(a + deriv_step) - mse(a)) / deriv_step
        a -= step_scale * deriv
        if abs(deriv) <= stop_deriv:
            break

    return {"a": a, "iterations": iteration, "converged": iteration < max_iter}

# --- Prediction from model ---
def predict_plm(model, newdata):
    assert "a" in model and "y0" in model
    a, y0 = model["a"], model["y0"]
    assert isinstance(newdata, np.ndarray)
    return y0 * newdata ** a

# --- Plotting power-law model ---
import matplotlib.pyplot as plt

def plot_plm(model, from_val, to_val, n=101, **kwargs):
    x = np.linspace(from_val, to_val, n)
    y = predict_plm(model, x)
    plt.plot(x, y, **kwargs)
    plt.title("Power-Law Model Plot")
    plt.xlabel("Predictor")
    plt.ylabel("Response")
    plt.grid(True)
    plt.show()
    return True

# --- Recursive Examples ---
def my_factorial(n):
    if n == 1:
        return 1
    return n * my_factorial(n - 1)

def fib(n):
    if n in (0, 1):
        return 1
    return fib(n - 1) + fib(n - 2)

print(my_factorial(5))  # Output: 120
print(fib(5))           # Output: 8