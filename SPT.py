import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MONPRC = pd.read_csv('D:/SPT.csv')
ticker = ['MMM', 'PG', 'WMT', 'XOM', 'C', 'AAPL']

NUM = len(ticker)    # no. of stocks
PED = int(MONPRC.shape[0]/NUM)   # no. of time periods
prc = np.zeros((PED, NUM))     # extract raw prices, not adjusted from stock split
share = np.zeros((PED, NUM))    # outstanding shares, in terms of 1000
for i in range(NUM):
    prc[:, i] = MONPRC.loc[MONPRC['TICKER'] == ticker[i]]['PRC'].values
    share[:, i] = MONPRC.loc[MONPRC['TICKER'] == ticker[i]]['SHROUT'].values

# market value, = price times shares
totmv = np.multiply(prc, share)
# mu (date, relative market weight)
mu = np.divide(totmv, np.tile(np.sum(totmv, axis=1), (NUM, 1)).T)
# window side used to estimate variance
T = 12

# additively strategy
portfolio = np.ones((PED-T+1, 1))
monrtn = np.zeros((PED-T, 1))
var = np.zeros((PED-T, NUM))
GammaH = np.zeros((PED-T, 1))
for t in range(PED-T):
    var[t, :] = np.var(mu[t:t+T, :], axis=0)
integrand = np.divide(var, mu[T-1:PED-1, :])   # integrand for Gamma^H, sigma^2/mu.
for t in range(PED-T):
    GammaH[t] = 0.5*np.sum(integrand[:t+1, :])
# pi (date, stock), starting from month 12, (we have month 0)
pi = np.multiply(mu[T-1:PED-1, :], -np.log(mu[T-1:PED-1, :]) + np.tile(GammaH, (1, NUM)))
# normalize row sum to one
pi = np.divide(pi, np.tile(np.sum(pi, axis=1), (NUM, 1)).T)

# multiplicatively strategy
PIportfolio = np.ones((PED-T+1, 1))
c = 2
PImonrtn = np.zeros((PED-T, 1))
PI = np.multiply(mu[T-1:PED-1, :], -np.log(mu[T-1:PED-1, :]) + c)
# normalize row sum to one
PI = np.divide(PI, np.tile(np.sum(PI, axis=1), (NUM, 1)).T)

# rebalanced strategy
RBportfolio = np.ones((PED-T+1, 1))
RBmonrtn = np.zeros((PED-T, 1))

# testing
for t in range(PED-T):
    stkrtn = np.zeros((NUM, 1))
    for i in range(NUM):
        if share[t+T, i]/share[t+T-1, i] < 1.1 and share[t+T, i]/share[t+T-1, i] > 0.9:
            # no stock split happens, repurchase may happen, strictly speaking, maybe not correct.
            stkrtn[i] = prc[t+T, i]/prc[t+T-1, i] - 1
        else:
            stkrtn[i] = totmv[t+T, i]/totmv[t+T-1, i] - 1
    portfolio[t+1] = portfolio[t]*(1 + np.dot(pi[t, :], stkrtn))
    monrtn[t] = np.dot(pi[t, :], stkrtn)
    PIportfolio[t+1] = PIportfolio[t]*(1 + np.dot(PI[t, :], stkrtn))
    PImonrtn[t] = np.dot(PI[t, :], stkrtn)
    RBportfolio[t+1] = RBportfolio[t]*(1 + np.sum(stkrtn)/NUM)
    RBmonrtn[t] = np.sum(stkrtn)/NUM
print('Sharpe ratios of Additively, Mutiplicatively, Rebalanced strategy are %f, %f, and %f.\n'
      % (np.mean(monrtn)/np.std(monrtn)*np.sqrt(12), np.mean(PImonrtn)/np.std(PImonrtn)*np.sqrt(12),
         np.mean(RBmonrtn)/np.std(RBmonrtn)*np.sqrt(12)))
print('Annualized returns of Additively, Mutiplicatively, Rebalanced strategy are %f, %f, and %f.\n'
      % (np.power(portfolio[PED-T], 12/(PED-T))-1, np.power(PIportfolio[PED-T], 12/(PED-T))-1,
         np.power(RBportfolio[PED-T], 12/(PED-T))-1))

# plot the comparison figure
plt.plot(portfolio, label='Additive')
plt.plot(PIportfolio, label='Multiplicative')
plt.plot(RBportfolio, label='RB')
plt.legend(loc='best')
plt.title('Comparison')
# plt.savefig('compare.png', dpi=300)
plt.show()

# market value of each stock, in term of billion
# plt.figure(2, figsize=(8, 6), dpi=80)
# for idx in range(NUM):
#     ax = plt.subplot(3, 2, idx+1)
#     ax.set_title(ticker[idx])
#     ax.plot(totmv[:,idx]/1000000)
# plt.tight_layout()
# plt.show()

# proportion invested in each stock, additive strategy
# plt.figure(3, figsize=(8, 6), dpi=80)
# for idx in range(NUM):
#     ax = plt.subplot(3, 2, idx+1)
#     ax.set_title(ticker[idx])
#     ax.plot(pi[:,idx])
# plt.tight_layout()
# plt.show()
