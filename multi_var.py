import pandas as pd
import numpy as np
from math import sqrt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

input_names  = [
    "Median Listing Price (US dollars)",
    "Median household income (inflation-adjusted US dollars)",
    "Total Population"
]
output = "Total housing units"

source = 'resources/sea_uni.csv'

df = pd.read_csv(source).dropna()

x = df[[i_name for i_name in input_names]]
y = df[output]

regression = LinearRegression()
regression.fit(x.values, y)
print(regression.score(x.values, y))

print(regression.coef_)
print(regression.intercept_)

print(regression.predict(np.array([3816063.72371, 478425.953833]).reshape(1, -1)))