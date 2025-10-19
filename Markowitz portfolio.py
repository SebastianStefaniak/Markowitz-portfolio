import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns


#0) Zamknięcie poprzednich figur (czyszczenie pamięci)

plt.close('all')


#1) Pobranie danych

tickers = ['AAPL', 'MSFT', 'SPY', 'GLD', 'BTC-USD']
start = '2022-01-01'
end = '2025-01-01'

data = yf.download(tickers, start=start, end=end)['Close']
data = data.dropna()

#2) logarytmiczne dzienne zwroty
log_returns = np.log(data / data.shift(1)).dropna()

#3) Przygotowanie DataFrame na wyniki
num_portfolios = 500
results = pd.DataFrame(columns=['Ryzyko', 'Zwrot', 'Sharpe'])

mean_returns = log_returns.mean() * 252  # średni roczny zwrot
cov_matrix = log_returns.cov() * 252     # roczna macierz kowariancji


#4)  Symulacja portfeli

for i in range(num_portfolios):
    # losowe wagi
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)  # normalizacja, suma wag = 1

    #5) oczekiwany zwrot portfela
    portfolio_return = np.dot(weights, mean_returns)

    #6) odchylenie standardowe portfela (ryzyko)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


    #7)Sharpe ratio
    sharpe_ratio = portfolio_return / portfolio_volatility

    #8) dodanie wiersza do DataFrame
    results.loc[i] = [portfolio_volatility, portfolio_return, sharpe_ratio]


#9)  Wizualizacja Efficient Frontier

plt.figure(figsize=(10,6))
plt.scatter(
    results['Ryzyko'],
    results['Zwrot'],
    c=results['Sharpe'],
    cmap='viridis',
    alpha=0.8
)

#10) portfel o najwyższym Sharpe ratio
max_sharpe_idx = results['Sharpe'].idxmax()
plt.scatter(
    results.loc[max_sharpe_idx, 'Ryzyko'],
    results.loc[max_sharpe_idx, 'Zwrot'],
    c='red',
    s=100,
    label='Najwyższy Sharpe'
)

plt.title("Efektywna Granica (Efficient Frontier)")
plt.xlabel('Ryzyko (odchylenie standardowe)')
plt.ylabel('Oczekiwany zwrot roczny')
plt.colorbar(label='Sharpe Ratio')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.9)
plt.show()



#11) Wypisanie najlepszego portfela

print("Najlepszy portfel (maksymalne Sharpe Ratio):")
print(f"Ryzyko: {results.loc[max_sharpe_idx, 'Ryzyko']:.2%}")
print(f"Zwrot: {results.loc[max_sharpe_idx, 'Zwrot']:.2%}")
print(f"Sharpe Ratio: {results.loc[max_sharpe_idx, 'Sharpe']:.2f}")

