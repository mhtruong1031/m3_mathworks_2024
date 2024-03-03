import pandas as pd
from math import sqrt

def _sum(arr):
 
    # initialize a variable
    # to store the sum
    # while iterating through
    # the array later
    sum = 0
 
    # iterate through the array
    # and add each element to the sum variable
    # one at a time
    for i in arr:
        sum = sum + i
 
    return(sum)

x = pd.Series([612916
,624681
,637850
,653017
,668849])
y = pd.Series([306694
,309205
,311286
,315950
,322795])

def conf_int(x_pred, t_crit, x: pd.Series, y: pd.Series, slope):
    SS_total = _sum([item**2 for item in y]) - 1/len(x)*y.sum()**2
    SS_residual = slope*(x.multiply(y).sum() - 1/len(x) * x.sum() * y.sum())
    SS_Error = SS_total - SS_residual
    S_esp = sqrt(SS_Error/(len(x)-2))
    big_boi = sqrt(1 + 1/len(x) + (x_pred - x.mean())**2/(_sum([item**2 for item in x]) - 1/len(x)*x.sum()**2))
    return(t_crit * S_esp*big_boi)

x_pred = 540314.495987
print(x_pred - conf_int(68927, 2.4469, x, y, 0.280821), x_pred + conf_int(68927, 2.4469, x, y, 0.280821))