### Is this think working?
import numpy as np

# Takes spot price, strike price, discount rate, length of time, # of periods, dividends, and sigma (volatility)
# Returns the European pricing of a call or put option

def call_payoff(spot,strike):
    return np.maximum(0,spot-strike)
    
def put_payoff(spot,strike):
    return np.maximum(0,strike-spot)
    
def future_stocks(spot,strike,u,d,time,nper):
    return np.array([spot*u**(nper-i)*d**i for i in range(nper+1)])

def future_payoffs(spot,strike,u,d,time,nper,option='call'):
    if option == 'call':
        return call_payoff(future_stocks(spot,strike,u,d,time,nper),strike)
    elif option == 'put':
        return put_payoff(future_stocks(spot,strike,u,d,time,nper),strike)
    else:
        return None

def get_u(rate,div,time,sigma):
    return np.exp((rate-div)*time*sigma+sigma*np.sqrt(time))

def get_d(rate,div,time,sigma):
    return np.exp((rate-div)*time*sigma-sigma*np.sqrt(time))

def value(c_u, c_d, u, d, rate, time, kind = 'european', option = 'call') -> float:
    vals = np.exp(-rate*time)*(c_u*(np.exp(rate*time)-d)+c_d*(u-np.exp(rate*time)))/(u-d)
    if kind == 'american':
        vals = np.maximum(future_payoffs(spot,strike,u,d,time,vals.shape[0]-1,option=option),vals)
    if vals.shape[0] == 1:
        return vals[0]
    else:
        return value(vals[:-1],vals[1:],u,d,rate,time,kind=kind,option=option)

def european_binomial_pricer(spot,strike,rate,time,nper,div,sigma, option = 'call') -> float:
    kind = 'european'
    u = get_u(rate,div,time,sigma)
    d = get_d(rate,div,time,sigma)
    payoffs = future_payoffs(spot,strike,u,d,time,nper,option=option)
    return value(payoffs[:-1],payoffs[1:],u,d,rate,time,kind=kind,option=option)
    
    