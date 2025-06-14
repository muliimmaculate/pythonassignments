import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, binom
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Histogram
x = np.random.normal(loc=5000, scale=1000, size=50)
plt.hist(x, bins=8)
plt.xlabel("Income")
plt.title("Histogram of State Income in 1977")
plt.show()

# Earthquake histogram
y = np.random.uniform(0, 700, 1000)
plt.hist(y, bins=np.arange(0, 701, 70))
plt.xlabel("Earthquake Depth")
plt.title("Histogram of Earthquake Depths")
plt.show()

# ECDF
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data)+1) / len(data)
    return x, y

x_ecdf, y_ecdf = ecdf(x)
plt.step(x_ecdf, y_ecdf, where='post')
plt.xlabel("Income")
plt.title("ECDF of State Income in 1977")
plt.show()

y_ecdfx, y_ecdfy = ecdf(y)
plt.step(y_ecdfx, y_ecdfy, where='post')
plt.xlabel("Earthquake Depth")
plt.title("ECDF of Earthquake Depths")
plt.show()

# QQ Plot
sm.qqplot(x, line='45', color='red')
plt.title("QQ Plot - Income")
plt.show()

sm.qqplot(y, line='45', color='red')
plt.title("QQ Plot - Earthquake Depth")
plt.show()

# Boxplot
df = sns.load_dataset("tips")
sns.boxplot(x="day", y="total_bill", data=df)
plt.title("Boxplot of Total Bill by Day")
plt.show()

# Scatter plot
df_quakes = pd.DataFrame({
    "lat": np.random.uniform(-40, -30, 100),
    "long": np.random.uniform(170, 180, 100),
    "mag": np.random.uniform(4.0, 6.0, 100)
})
plt.scatter(df_quakes["long"], df_quakes["lat"])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Location of Earthquake Epicenters")
plt.show()

# Symbol plot
plt.scatter(df_quakes["long"], df_quakes["lat"], s=10 ** df_quakes["mag"])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Earthquake Magnitudes")
plt.show()

# Matrix plot (pairs)
sns.pairplot(sns.load_dataset("iris"))
plt.show()

# Pie chart
labels = ["Blueberry", "Cherry", "Apple", "Boston Creme", "Other", "Vanilla Creme"]
sizes = [0.12, 0.30, 0.26, 0.16, 0.04, 0.12]
colors = ["blue", "red", "green", "wheat", "orange", "white"]
plt.pie(sizes, labels=labels, colors=colors)
plt.title("Pie Chart Example")
plt.show()

# Barplot
VADeaths = np.random.randint(1, 10, (5, 4))
df_vadeaths = pd.DataFrame(VADeaths, columns=["Rural Male", "Rural Female", "Urban Male", "Urban Female"])
df_vadeaths.plot(kind="bar", stacked=False)
plt.title("Virginia Death Rates per 1000 in 1940")
plt.legend()
plt.show()

# Time series plot
ts_data = pd.Series(np.random.randint(100, 300, 144), index=pd.date_range("1949-01", periods=144, freq='M'))
ts_data.plot(title="International Airline Passengers")
plt.xlabel("Date")
plt.ylabel("Passengers (in thousands)")
plt.show()

# Binomial distribution
x_vals = np.arange(0, 6)
y_vals = binom.pmf(x_vals, 5, 0.4)
plt.vlines(x_vals, 0, y_vals)
plt.title("Binomial Distribution")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.show()

# Normal distribution
x = np.linspace(-3, 3, 600)
y = norm.pdf(x)
plt.plot(x, y)
plt.title("Normal Distribution")
plt.ylabel("f(x)")
plt.show()

# ECDF comparison
treated = np.random.normal(120, 10, 50)
untreated = np.random.normal(110, 10, 50)
xt, yt = ecdf(treated)
xu, yu = ecdf(untreated)
plt.step(xt, yt, label="Treated", where="post")
plt.step(xu, yu, label="Untreated", where="post", color="blue")
plt.legend()
plt.title("Treated vs Untreated")
plt.show()

# Save plot
plt.figure()
plt.step(xt, yt, label="Treated", where="post")
plt.step(xu, yu, label="Untreated", where="post", color="blue")
plt.legend()
plt.title("Treated vs Untreated")
plt.savefig("2cdfs.pdf")
plt.close()

# Multiple plot on one figure
x = np.linspace(0, 2*np.pi, 100)
sine = np.sin(x)
cosine = np.cos(x)
plt.plot(x, sine, label="Sine")
plt.plot(x, cosine, label="Cosine")
plt.legend()
plt.title("Sine and Cosine")
plt.show()

# Multi-frame plotting
fig, axs = plt.subplots(2, 2)
axs[0, 0].boxplot(df["total_bill"])
axs[0, 1].hist(df["total_bill"])
axs[1, 0].plot(*ecdf(df["total_bill"]))
sm.qqplot(df["total_bill"], line='45', ax=axs[1, 1])
plt.tight_layout()
plt.show()