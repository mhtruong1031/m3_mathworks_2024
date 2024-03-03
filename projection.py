import pandas as pd
import numpy as np
from math import sqrt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

input_names  = [
    "Median Listing Price (US dollars)",
    "Median household income (inflation-adjusted US dollars)"
]
output = "Total housing units"

source = 'resources/alb_uni.csv'

df = pd.read_csv(source).dropna()

x = np.array(df["Median household income (inflation-adjusted US dollars)"].tolist()).reshape(-1, 1)
y = df["Total Population"]

regression = LinearRegression()
regression.fit(x, y)
print(regression.score(x, y))

print(regression.coef_)
print(regression.intercept_)
print(regression.predict(np.array([912008.043956044])).reshape(-1, 1))

