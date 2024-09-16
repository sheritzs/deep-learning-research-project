
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import STL


def correlation_matrix(df, figsize=(18,6), cmap='coolwarm', mask=True, name=None):
    """Accepts a dataframe and generates a correlation matrix. If
    a name is provided, the image is saved."""
    corr = df.corr()

    plt.subplots(figsize=figsize)

    # generate a mask for the upper triangle
    if mask:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        heatmap = sns.heatmap(corr, mask=mask, vmax=1, vmin=-1,
            annot=True, cmap=cmap)
    else:
        heatmap = sns.heatmap(corr, vmax=1, vmin=-1,
            annot=True, cmap=cmap)

    plt.tight_layout()

    if name:
        plt.savefig(f'{name}.png')

    plt.show()

def plot_seasonal_decomposition(df, column, period, color, name=''):
    """Provides styled decomposition plots for a given dataframe and column."""
    decomposition = STL(df[column], period=period).fit()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True,
                                        figsize=(10,8))

    ax1.plot(decomposition.observed, color=color)
    ax1.set_ylabel('Observed')

    ax2.plot(decomposition.trend, color=color)
    ax2.set_ylabel('Trend')

    ax3.plot(decomposition.seasonal, color=color)
    ax3.set_ylabel('Seasonal')

    ax4.plot(decomposition.resid, color=color)
    ax4.set_ylabel('Residuals')

    fig.autofmt_xdate()
    plt.tight_layout()

    if name:
        plt.savefig(f'{name}.png')

    plt.show()


