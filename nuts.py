import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
regression = LinearRegression()

def get_uni__results(input: pd.Series, output: pd.Series) -> float:
    x = np.array([float(val) for val in input.tolist()]).reshape((-1, 1))
    y = np.array([float(val) for val in output.tolist()]).reshape((-1, 1))
    

    regression.fit(x, y)

    return regression.score(x, y)

source = 'resources/sea_uni.csv'

df = pd.read_csv(source)

var1 = df["Median Listing Price (US dollars)"]
var2 = df["Median household income (inflation-adjusted US dollars)"]
var3 = df["Total Population"]

print(get_uni__results(var1, var2))

print(get_uni__results(var2, var3))

print(get_uni__results(var1, var3))

