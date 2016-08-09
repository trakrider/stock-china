"""Market Simulator"""


""" Accept trades and simulator the portfolio
    Output daily portfolio values
"""

import numpy as np

def simulate(price, trades, starting_cash = 0):
    cash = np.ones(price.shape[0]) * starting_cash # Track cash
    qty = np.zeros(price.shape[0])                 # Track quantity
    trade_not_existe = (trades == 0)
    for i in range(price.shape[0]):
        if trade_not_existe[i] == False:
            # Execute trade
            cash[i:] -= trades[i]*price[i]  # Update cash
            qty[i:]  += trades[i]           # Update position qty
    port_val = np.zeros(price.shape[0])
    port_val = price*qty + cash
    return port_val
