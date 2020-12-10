
import numpy as np
from scipy.stats import norm

# Takes spot price, strike price, discount rate, length of time, # of periods, dividends, and sigma (volatility)
# Returns the European pricing of a call or put option

def european_binomial_pricer(spot,strike,rate,time,nper,div,sigma,option='call') -> float:
    kind = 'european'
    u = get_u(rate,time,div,sigma)
    d = get_d(rate,time,div,sigma)
    payoffs = future_payoffs(spot,strike,u,d,time,nper,option=option)
    return value(payoffs[:-1],payoffs[1:],u,d,rate,time,kind=kind,option=option)

def value(c_u, c_d, u, d, rate, time, kind = 'european', option = 'call') -> float:
    vals = np.exp(-rate*time)*(c_u*(np.exp(rate*time)-d)+c_d*(u-np.exp(rate*time)))/(u-d)
    if kind == 'american':
        vals = np.maximum(future_payoffs(spot,strike,u,d,time,vals.shape[0]-1,option=option),vals)
    if vals.shape[0] == 1:
        return vals[0]
    elif vals.shape[0] == 0:
        return None
    else:
        return value(vals[:-1],vals[1:],u,d,rate,time,kind=kind,option=option)

def future_payoffs(spot,strike,u,d,time,nper,option='call'):
    if option == 'call':
        return call_payoff(future_stocks(spot,strike,u,d,time,nper),strike)
    elif option == 'put':
        return put_payoff(future_stocks(spot,strike,u,d,time,nper),strike)
    else:
        return None
    
def future_stocks(spot,strike,u,d,time,nper):
    return np.array([spot*u**(nper-i)*d**i for i in range(nper+1)])

def call_payoff(spot,strike):
    return np.maximum(0,spot-strike)
    
def put_payoff(spot,strike):
    return np.maximum(0,strike-spot)

def get_u(rate,time,div,sigma):
    return np.exp((rate-div)*time*sigma+sigma*np.sqrt(time))

def get_d(rate,time,div,sigma):
    return np.exp((rate-div)*time*sigma-sigma*np.sqrt(time))

    
def black_scholes_pricer(spot,strike,rate,time,div,sigma,option='call'):
    d1 = get_d1(spot,strike,rate,time,div,sigma)
    d2 = get_d2(d1,time,sigma)
    fn = bs_call if option == 'call' else bs_put
    return fn(spot,strike,d1,d2,rate,time,div,sigma)

def bs_call(spot,strike,d1,d2,rate,time,div,sigma):
    return spot*np.exp(-div*time)*norm.cdf(d1)-strike*np.exp(-rate*time)*norm.cdf(d2)

def bs_put(spot,strike,d1,d2,rate,time,div,sigma):
    return strike*np.exp(-rate*time)*norm.cdf(-d2)-spot*np.exp(-div*time)*norm.cdf(-d1)

def get_d1(spot,strike,rate,time,div,sigma):
    return (np.log(strike/spot) + (rate-div+.5*sigma**2)*time)/(sigma*np.sqrt(time))

def get_d2(d1,time,sigma):
    return d1-sigma*np.sqrt(time)