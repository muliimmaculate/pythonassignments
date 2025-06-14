# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split

# Load dataset
url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/birthwt.csv"
birthwt = pd.read_csv(url).drop(columns="Unnamed: 0")

# Rename columns
birthwt.columns = [
    "birthwt.below.2500", "mother.age", "mother.weight", "race",
    "mother.smokes", "previous.prem.labor", "hypertension", "uterine.irr",
    "physician.visits", "birthwt.grams"
]

# Recode categorical variables
birthwt["race"] = birthwt["race"].map({1: "white", 2: "black", 3: "other"}).astype("category")
birthwt["mother.smokes"] = birthwt["mother.smokes"].map({0: "No", 1: "Yes"}).astype("category")
birthwt["hypertension"] = birthwt["hypertension"].map({0: "No", 1: "Yes"}).astype("category")
birthwt["uterine.irr"] = birthwt["uterine.irr"].map({0: "No", 1: "Yes"}).astype("category")

# Plot examples
birthwt["race"].value_counts().plot(kind="bar", title="Mother's Race")
plt.show()

birthwt["mother.age"].plot(kind="hist", title="Mother's Age Distribution")
plt.show()

birthwt.sort_values("mother.age")["mother.age"].plot(title="Sorted Mother's Age")
plt.show()

plt.scatter(birthwt["mother.age"], birthwt["birthwt.grams"])
plt.title("Birth Weight by Mother's Age")
plt.xlabel("Mother's Age")
plt.ylabel("Birth Weight (g)")
plt.show()

# T-Test
yes = birthwt[birthwt["mother.smokes"] == "Yes"]["birthwt.grams"]
no = birthwt[birthwt["mother.smokes"] == "No"]["birthwt.grams"]
print(ttest_ind(yes, no))

# Linear models
lm1 = smf.ols("birthwt.grams ~ mother.smokes", data=birthwt).fit()
print(lm1.summary())

lm2 = smf.ols("birthwt.grams ~ mother.age", data=birthwt).fit()
print(lm2.summary())

# Remove outliers
birthwt_noout = birthwt[birthwt["mother.age"] <= 40]

lm3 = smf.ols("birthwt.grams ~ mother.age", data=birthwt_noout).fit()
print(lm3.summary())

lm3a = smf.ols("birthwt.grams ~ mother.smokes + mother.age", data=birthwt_noout).fit()
print(lm3a.summary())

lm3b = smf.ols("birthwt.grams ~ mother.age + mother.smokes * race", data=birthwt_noout).fit()
print(lm3b.summary())

# Full model excluding dependent-derived column
lm4a = smf.ols("birthwt.grams ~ mother.age + mother.weight + race + mother.smokes + "
               "previous.prem.labor + hypertension + uterine.irr + physician.visits",
               data=birthwt_noout).fit()
print(lm4a.summary())

# Generalized Linear Model
glm1 = smf.glm("birthwt.below.2500 ~ mother.age + mother.weight + race + mother.smokes + "
               "previous.prem.labor + hypertension + uterine.irr + physician.visits",
               data=birthwt_noout, family=sm.families.Binomial()).fit()
print(glm1.summary())

# Split into train/test
birthwt_in = birthwt_noout.iloc[::2]
birthwt_out = birthwt_noout.iloc[1::2]

lm_half = smf.ols("birthwt.grams ~ mother.age + mother.weight + race + mother.smokes + "
                  "previous.prem.labor + hypertension + uterine.irr + physician.visits",
                  data=birthwt_in).fit()
print(lm_half.summary())

train_pred = lm_half.predict(birthwt_in)
test_pred = lm_half.predict(birthwt_out)

# Correlation
print("Train correlation:", np.corrcoef(birthwt_in["birthwt.grams"], train_pred)[0, 1])
print("Test correlation:", np.corrcoef(birthwt_out["birthwt.grams"], test_pred)[0, 1])

# Predictions plot
plt.scatter(birthwt_in["birthwt.grams"], train_pred)
plt.title("Train Predictions")
plt.show()

plt.scatter(birthwt_out["birthwt.grams"], test_pred)
plt.title("Test Predictions")
plt.show()