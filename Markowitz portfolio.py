import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

#0) Close previous figures (memory cleanup)
plt.close('all')

#1) Download data
tickers = ['AAPL', 'MSFT', 'SPY', 'GLD', 'BTC-USD']
start = '2022-01-01'
end = '2025-01-01'
data = yf.download(tickers, start=start, end=end)['Close']
data = data.dropna()

#2) logarithmic daily returns
log_returns = np.log(data / data.shift(1)).dropna()

#3) Prepare results DataFrame
num_portfolios = 500
results = pd.DataFrame(columns=['Risk', 'Return', 'Sharpe'])
mean_returns = log_returns.mean() * 252  # average annual return
cov_matrix = log_returns.cov() * 252  
# annual covariance matrix
#4)  Portfolio simulation
for i in range(num_portfolios):
    # random weights
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)  # normalization, weights sum to 1
    #5) expected portfolio return
    portfolio_return = np.dot(weights, mean_returns)
    #6) portfolio standard deviation (risk)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    #7)Sharpe ratio
    sharpe_ratio = portfolio_return / portfolio_volatility
    #8) add row to DataFrame
    results.loc[i] = [portfolio_volatility, portfolio_return, sharpe_ratio]
    
#9)  Efficient Frontier visualization
plt.figure(figsize=(10,6))
plt.scatter(
    results['Risk'],
    results['Return'],
    c=results['Sharpe'],
    cmap='viridis',
    alpha=0.8
)
#10) portfolio with the highest Sharpe ratio
max_sharpe_idx = results['Sharpe'].idxmax()
plt.scatter(
    results.loc[max_sharpe_idx, 'Risk'],
    results.loc[max_sharpe_idx, 'Return'],
    c='red',
    s=100,
    label='Highest Sharpe'
)
plt.title("Efficient Frontier")
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Annual Return')
plt.colorbar(label='Sharpe Ratio')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.9)
plt.show()
#11) Print best portfolio
print("Best portfolio (maximum Sharpe Ratio):")
print(f"Risk: {results.loc[max_sharpe_idx, 'Risk']:.2%}")
print(f"Return: {results.loc[max_sharpe_idx, 'Return']:.2%}")
print(f"Sharpe Ratio: {results.loc[max_sharpe_idx, 'Sharpe']:.2f}")
 
