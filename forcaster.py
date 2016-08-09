""" Forecaster!
"""
import math
import talib
import numpy as np
import pandas as pd
import KNNLearner as knn
from util import get_stock, fill_missing_values

def run_forcaster(dates, symbol, predict_periods, last_date_in_sample = '2015-12-31'):

    # Get stock price
    df = get_stock(symbol, dates)   # Get adjusted closing price from csv file
    # Fill NA
    fill_missing_values(df)
    # Compute point return as Y
    df['rtn'] = df['close'].pct_change(periods = predict_periods)
    df['real_rtn_forward'] = df['rtn'].shift(periods = -predict_periods)
    # Compute X - indicators
    df['indicator_bb'] = compute_indicator_bb(df['close'], window = 20)
    df['indicator_volatility'] = compute_indicator_volatility(df['close'])
    df['indicator_momentum'] = compute_indicator_momentum(df['close'])
    df['indicator_macd'] = compute_indicator_MACD(df['close'])
    df['indicator_stoch'] = compute_indicator_stoch(df)
    # Normalize X
    df1 = df.ix[:-5].dropna(axis = 0)
    df2 = df.ix[-5:]
    df = pd.concat([df1,df2])
    df[['indicator_stoch','indicator_momentum','indicator_macd','indicator_volatility','indicator_bb']] = 100*df[['indicator_stoch','indicator_momentum','indicator_macd','indicator_volatility','indicator_bb']]/df[['indicator_stoch','indicator_momentum','indicator_macd','indicator_volatility','indicator_bb']].std()


    # Grab X and Y
    trainX = df[['indicator_bb', 'indicator_momentum', 'indicator_volatility', 'indicator_stoch', 'indicator_macd']][:last_date_in_sample].values
    trainY = df['real_rtn_forward'][:last_date_in_sample].values
    testX_insample = trainX
    testY_insample = trainY
    testX_outsample = df[['indicator_bb', 'indicator_momentum', 'indicator_volatility', 'indicator_stoch', 'indicator_macd']]['2016-07-01':].values
    testY_outsample = df['real_rtn_forward']['2016-07-01':].values
    # Train a learner
    learner = knn.KNNLearner(k=3)
    learner.addEvidence(trainX, trainY)
    # Throw input to the learner
    predY_insample  = learner.query(testX_insample)
    predY_outsample = learner.query(testX_outsample)
    # Evaluation
    eval_insample  = evaluateLearnerOutput(predY_insample, testY_insample)
    eval_outsample = evaluateLearnerOutput(predY_outsample, testY_outsample)
    # Generate Y
    df['pred_rtn_forward'] = np.concatenate((predY_insample, predY_outsample))
    return df,eval_insample,eval_outsample

def compute_indicator_bb(ds_price, window = 20, is_needplot = False):
    rolling_mean = ds_price.rolling(window = window, center = False).mean()
    rolling_std  = ds_price.rolling(window = window, center = False).std()
    if is_needplot == True:
        # Create band
        upper_band = rolling_mean + 2*rolling_std
        lower_band = rolling_mean - 2*rolling_std
        # Create plot
        ax = ds_price.plot(title="Bollinger Bands")
        rolling_mean.plot(label='Rolling mean', ax=ax)
        upper_band.plot(label='upper band', ax=ax)
        lower_band.plot(label='lower band', ax=ax)
        # Add axis labels and legend
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='upper left')
        plt.show()
    bb_value = (ds_price - rolling_mean)/(2 * rolling_std)
    return bb_value

def compute_indicator_momentum(ds_price):
    return ds_price.pct_change(periods=10)

def compute_indicator_volatility(ds_price, timeperiod=5):
    '''
    Calculate volatility for the previous [timeperiod] days
    '''
    df = pd.DataFrame(ds_price)
    df['daily_rtn'] = ds_price.pct_change(periods=1)
    df['volatility'] = df['daily_rtn'].rolling(window = timeperiod, center = False).std()
    return df['volatility']



def compute_indicator_MACD(ds_price, fastperiod=8, fastmatype=1, slowperiod=17, slowmatype=1, signalperiod=9, signalmatype=1):
    '''
    Calculate MACD

    optional args:
    #   list of values for the Moving Average Type:
    #0: SMA (simple)
    #1: EM  (exponential)
    #2: WMA (weighted)
    #3: DEMA (double exponential)
    #4: TEMA (triple exponential)
    #5: TRIMA (triangular)
    #6: KAMA (Kaufman adaptive)
    #7: MAMA (Mesa adaptive)
    #8: T3 (triple exponential T3)
    '''
    macd, signal, hist = talib.MACDEXT(ds_price.values,
                                       fastperiod=fastperiod,
                                       slowperiod=slowperiod,
                                       signalperiod=signalperiod,
                                       fastmatype=fastmatype,
                                       slowmatype=slowmatype,
                                       signalmatype=signalmatype)
    return hist

def compute_indicator_stoch(df):
    '''
    Calculate stoch
    Input: df[['close', 'high', 'low']]
    '''
    df['k'], df['d'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values, fastk_period=14, slowk_period=1, slowd_period=5)
    return df['d']

def compute_indicator_test1(ds_price):
    '''
    Calculate Momentum: Price - SMA(30)
    Return both the momentum and D3(momentum)

    '''
    df = pd.DataFrame(ds_price)
    df['sma30d'] = talib.SMA(ds_price.values, timeperiod=30)
    df['momentum30d'] = ds_price - df['sma30d']
    df['momentum30d_shift3'] = df['momentum30d'].shift(periods = 3)
    df['momentum30d_delta3'] = df['momentum30d'] - df['momentum30d_shift3']
    return df[['momentum30d', 'momentum30d_delta3']]



def evaluateLearnerOutput(predY, realY):
    result = dict()
    c = np.corrcoef(predY, y=realY)
    result['rmse'] = math.sqrt(((realY - predY) ** 2).sum()/realY.shape[0])
    result['corr'] = c[0,1]
    return result
