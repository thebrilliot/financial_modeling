import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt
import random

# Takes spot price, strike price, discount rate, length of time, # of periods, dividends, and sigma (volatility)
# Returns the European pricing of a call or put option

# I was using a recursive function before I started, but this ended up being easier
def european_binomial_pricer(spot, strike, expiry, rate, div, vol, num: int, option='call') -> float:
    h = expiry/num
    u = get_u(rate,h,div,vol)
    d = get_d(rate,h,div,vol)
    vals = future_payoffs(spot,strike,u,d,num,option=option)
    while vals.shape[0] > 1:
        vals = value(spot,vals[:-1],vals[1:],u,d,h,rate,div)
    return vals[0]

def american_binomial_pricer(spot, strike, expiry, rate, div, vol, num: int, option='call') -> float:
    h = expiry/num
    u = get_u(rate,h,div,vol)
    d = get_d(rate,h,div,vol)
    vals = future_payoffs(spot,strike,u,d,num,option=option)
    while vals.shape[0] > 1:
        vals = value(spot,vals[:-1],vals[1:],u,d,h,rate,div)
        exercise = future_payoffs(spot,strike,u,d,vals.shape[0]-1,option=option)
        vals = np.maximum(vals, exercise)
    return vals[0]

# This is the magic function finding the value of options at a given node, works with entire layers of the binomial tree
# I have both the Delta/B method and the exact method and I leave one commented
def value(spot, c_u, c_d, u, d, h, rate, div) -> float:
    #Delta*Spot+B
    #delta = get_delta(spot,c_u,c_d,u,d,h,div)
    #b = get_b(c_u,c_d,u,d,h,rate)
    #vals = delta*spot+b
    
    #Risk-neutral format
    p_star = get_p_star(u,d,h,rate,div)
    vals = np.exp(-rate*h)*(c_u*p_star+c_d*(1-p_star))
    
    #Exact equation
    #vals = np.exp(-rate*h)*(c_u*(np.exp((rate-div)*h)-d)+c_d*(u-np.exp((rate-div)*h)))/(u-d)
    return vals

def get_p_star(u,d,h,rate,div):
    return (np.exp((rate-div)*h)-d)/(u-d)

def get_delta(spot,c_u,c_d,u,d,h,div):
    return np.exp(-div*h)*((c_u-c_d)/(spot*(u-d)))

def get_b(c_u,c_d,u,d,h,rate):
    return np.exp(-rate*h)*(u*c_d-d*c_u)/(u-d)

# Future option payoffs
def future_payoffs(spot,strike,u,d,num,option='call'):
    if option == 'call':
        return call_payoff(future_stocks(spot,u,d,num),strike)
    elif option == 'put':
        return put_payoff(future_stocks(spot,u,d,num),strike)
    else:
        return None
    
# Future stock prices
def future_stocks(spot,u,d,num) -> np.ndarray:
    return np.array([spot*u**(num-i)*(d**i) for i in range(num+1)])

def call_payoff(spot,strike):
    return np.maximum(0,spot-strike)
    
def put_payoff(spot,strike):
    return np.maximum(0,strike-spot)

def get_u(rate,h,div,vol):
    return np.exp((rate-div)*h+vol*np.sqrt(h))

def get_d(rate,h,div,vol):
    return np.exp((rate-div)*h-vol*np.sqrt(h))


def black_scholes_call(spot, strike, expiry, rate, div, vol) -> float:
    d1 = get_d1(spot, strike, expiry, rate, div, vol)
    d2 = get_d2(d1,expiry,vol)
    return (spot * np.exp(-div * expiry) * norm.cdf(d1)) - (strike * np.exp(-rate * expiry) * norm.cdf(d2))

def black_scholes_put(spot, strike, expiry, rate, div, vol) -> float:
    d1 = get_d1(spot,strike,expiry,rate,div,vol)
    d2 = get_d2(d1,expiry,vol)
    return (strike * np.exp(-rate * expiry) * norm.cdf(-d2)) - (spot * np.exp(-div * expiry) * norm.cdf(-d1))

def get_d1(spot, strike, expiry, rate, div, vol):
    return (np.log(strike/spot) + (rate - div + 0.5 * vol * vol) * expiry) / (vol * np.sqrt(expiry))

def get_d2(d1,expiry,vol):
    return d1 - vol * np.sqrt(expiry)



def binomial_path(spot,expiry,rate,num,div,vol):
    h = expiry/num
    u = get_u(rate,h,div,vol)
    d = get_d(rate,h,div,vol)
    updowns = random.choices([0,1],k=num)
    path = [spot]
    for move in updowns:
        path.append(path[-1]*u**move*d**(1-move))
    return np.array(path)


def parity(spot,strike,expiry,rate):
    return spot-strike*np.exp(-rate*expiry)
    #return (spot*np.exp(rate*expiry)-strike)*np.exp(-rate*expiry)

## Delta - I literally have no idea what this is for
    #def black_scholes_call_delta(spot: float, strike: float, tau: float, rate: float, div: float, vol: float) -> float:
    #d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * tau) / (vol * np.sqrt(tau))
    #return np.exp(-div * tau) * norm.cdf(d1)