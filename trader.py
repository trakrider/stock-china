"""Trader"""

import numpy as np

def generate_trade(rtn):
    qty = np.zeros(rtn.shape[0])
    rtn_is_larger_than_upper_level = rtn > 0.01
    # Write trade rules
    for i in range(rtn.shape[0] - 5):
        if rtn_is_larger_than_upper_level[i] == True:
            qty[i] = 100
            qty[i+5] = -100
            rtn_is_larger_than_upper_level[i+1:i+1+5] = False # Lock the trade for 5d
    return qty

# Test case for this python file
if __name__=="__main__":
    print("Command line execution not implemented")
