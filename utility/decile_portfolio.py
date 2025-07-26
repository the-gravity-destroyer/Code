import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DecilePortfolioAnalysis:
    """
    Durchführung und Visualisierung einer Decile-Portfolio-Analyse für Modellvorhersagen.
    """
    def __init__(self, n_stocks: int):
        self.n_stocks = n_stocks

    def compute_decile_returns(self, y_pred: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
        """
        Ordnet Vorhersagen in monatliche Deciles und berechnet die realisierten Returns je Decile.
        Gibt einen DataFrame mit mittlerem Return und Sharpe Ratio pro Decile zurück.
        """
        # Zeitliche Zuordnung (Monat à n_stocks Beobachtungen)
        months = np.arange(len(y_true)) // self.n_stocks
        unique_months = np.unique(months)

        # Sammelstruktur für Decile-Returns
        decile_returns = {dec: [] for dec in range(1, 11)}

        # Loop through months
        for m in unique_months:
            mask = months == m
            preds_m  = y_pred[mask]
            truths_m = y_true[mask]
            # Decile-Ranking (1..10)
            deciles = pd.qcut(preds_m, 10, labels=False) + 1
            for dec in range(1, 11):
                idx = deciles == dec
                if idx.any():
                    decile_returns[dec].append(truths_m[idx].mean())

        # Compute mean return & Sharpe per decile
        records = []
        for dec in range(1, 11):
            arr = np.array(decile_returns[dec])
            mean_ret = arr.mean()
            std_ret  = arr.std(ddof=1)
            sharpe   = mean_ret / std_ret if std_ret != 0 else np.nan
            records.append({'decile': dec, 'mean_return': mean_ret, 'std_return': std_ret, 'sharpe_ratio': sharpe})

        df = pd.DataFrame(records).set_index('decile')
        return df

    def plot_sharpe(self, df_deciles: pd.DataFrame, title: str = "Decile Sharpe Ratios") -> None:
        """
        Bar-Chart der Sharpe Ratios je Decile.
        """
        plt.figure(figsize=(8, 5))
        plt.bar(df_deciles.index, df_deciles['sharpe_ratio'])
        plt.xlabel('Decile')
        plt.ylabel('Sharpe Ratio')
        plt.title(title)
        plt.xticks(df_deciles.index)
        plt.tight_layout()
        plt.show()

    def plot_cumulative_returns(self, df_deciles: pd.DataFrame, title: str = "Decile Cumulative Returns") -> None:
        """
        Liniendiagramm der kumulativen Returns je Decile.
        """
        cumulative = (1 + df_deciles['mean_return']).cumprod() - 1
        plt.figure(figsize=(8, 5))
        plt.plot(cumulative.index, cumulative.values, marker='o')
        plt.xlabel('Decile')
        plt.ylabel('Cumulative Return')
        plt.title(title)
        plt.xticks(cumulative.index)
        plt.tight_layout()
        plt.show()
