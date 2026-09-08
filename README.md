# Efficient Frontier Simulation

This project demonstrates random portfolio simulations using the Monte Carlo method and visualizations of the efficient frontier based on actual historical market data. 

The simulation calculates the following for each portfolio:
- Expected annual rate of return
- Risk (annual standard deviation)
- Sharpe ratio
## 📊 Results
<img width="972" height="589" alt="image" src="https://github.com/user-attachments/assets/7e2ba97f-2390-4933-a909-f21b3fc5a121" />

Each point represents a randomly weighted portfolio of AAPL, MSFT, SPY, GLD, and BTC-USD (500 simulations), plotted by risk (volatility) against expected annual return and colored by Sharpe ratio. The portfolio with the highest Sharpe ratio (highlighted in red) sits toward the lower-risk end of the simulated range rather than at the highest absolute return 
- **Risk:** ~14.5%
- **Expected annual return:** ~12.5%
- **Sharpe Ratio:** highest among all 500 simulated portfolios

The goal of the project is to present a set of efficient portfolios offering the highest rate of return at a given level of risk and to select the one that offers the highest Sharpe ratio.
