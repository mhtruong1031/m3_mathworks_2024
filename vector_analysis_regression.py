import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base import datetools
from statsmodels.tsa.stattools import grangercausalitytests

mdata  = pd.read_csv('resources/vector_auto_reg/collective.csv')
yearly = mdata['Year'].astype(int).astype(str)
yearly = datetools.dates_from_str(yearly)

mata = mdata[['Homeless Total', 'FO Housing', 'SO Med Listing Price', 'FO Population', 'SO Poverty Perc']]
mdata.index = pd.DatetimeIndex(yearly)
mdata.head()

data   = mdata[['Homeless Total', 'FO Housing', 'SO Med Listing Price', 'FO Population', 'SO Poverty Perc']].pct_change().dropna()
#gc_res = grangercausalitytests(data, 1)

var = VAR(data)
x   = var.select_order()

results = var.fit(1)

#print(x.summary())
print(results.summary())
print(pd.DataFrame(results.forecast(data.values[results.k_ar:], 50)).to_csv('vec_autoreg_results.csv'))