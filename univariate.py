import pandas as pd
import numpy as np
from math import sqrt
from sklearn.linear_model import LinearRegression

regression = LinearRegression()

def get_uni__results(input: pd.Series, output: pd.Series) -> float:
    x = np.array([float(val) for val in input.tolist()]).reshape((-1, 1))
    y = np.array([float(val) for val in output.tolist()]).reshape((-1, 1))
    

    regression.fit(x, y)

    return regression.score(x, y)

source = 'resources/sea_uni.csv'

input_names  = [
    #"Total housing units",
    "Median Listing Price (US dollars)",
    "Total Population",
    "Median household income (inflation-adjusted US dollars)"
]
output_names = [
    #"Total Change in Supply",
    "Total housing units",
    #"Market Entry",
    #"Sold Listings"
]

df = pd.read_csv(source).dropna()
print(df)

for o_name in output_names:
    output = df[o_name]
    for i_name in input_names:
        input = df[i_name]
        print(i_name + " --> " + o_name)
        print(sqrt(get_uni__results(input, output)))

'''output = df["Total housing units"]
for col in df.columns[2:]:
    input = df[col]
    print(col)
    print(get_uni__results(input, output))
'''    
    
