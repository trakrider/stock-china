import datetime as dt
import pandas as pd
from forcaster import run_forcaster
from trader import generate_trade
from marketsimulator import simulate

""" Run back test.
    Input dates
    Input edge for in/out sample
    Input symbol
    Input predict periods
"""
def run_backtest():
    # input
    dates = pd.date_range(dt.datetime(2000,12,31), dt.datetime(2016,8,30))
    last_date_in_sample = '2016-6-30'
    symbol  = 'sh600820'
    predict_periods = 5
    # Evaluate learner
    df, eval_insample, eval_outsample = run_forcaster(dates, symbol, predict_periods, last_date_in_sample = last_date_in_sample)
    # Generate trades
    df['tradeQTY'] = generate_trade(df['pred_rtn_forward'].values)
    # Run market simulator and get daily port value
    df['port_val'] = simulate(df['close'].values, df['tradeQTY'].values, starting_cash = 10000)
    # Plot port_val
    df = df.ix[-100:]
    ax = df['port_val'].plot(title="Back Test for Portfolio Value", label='Port Val')
    ax.set_xlabel("Date")
    ax.set_ylabel("Port Value")
    ax.legend(loc='upper left')
    df[['pred_rtn_forward', 'real_rtn_forward']].plot(label='Prediction', ax=ax, secondary_y=True)
    fig = ax.get_figure()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('asdf.png',dpi=200)
    print (eval_insample)
    print (eval_outsample)
    print (df['pred_rtn_forward'][-5:]) # Print prediction about future return


if __name__=="__main__":
    run_backtest()
