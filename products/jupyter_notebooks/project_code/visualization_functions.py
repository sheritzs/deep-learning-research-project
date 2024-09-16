
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

def plot_monthly_charts(monthly_data: dict, column='sunshine_hr', 
                        figsize=(20,20), x_label='', y_label='Hours' ,
                        name=None):
    
    """Plots the Annual Trend, Monthly means, and Three Year Rolling Average."""
    
    YEAR = 'year'
    SUNSHINE_HR = 'sunshine_hr'
    THREE_YEAR_ROLLING_AVG = '3yr_rolling_avg'
    JANUARY = 'January'
    FEBRUARY = 'February'
    MARCH = 'March'
    APRIL = 'April'
    MAY = 'May'
    JUNE = 'June'
    JULY = 'July'
    AUGUST = 'August'
    SEPTEMBER = 'September'
    OCTOBER = 'October'
    NOVEMBER = 'November'
    DECEMBER = 'December'

    # Monthly Subsets

    y_max = float('-inf')

    for month in monthly_data.keys():
        monthly_max = monthly_data[month]['sunshine_hr'].max()
        if monthly_max > y_max:
            y_max = monthly_max

    y_max = np.ceil(y_max) # will use to set consistent upper limit for charts

    fig, axes = plt.subplots(4, 3, figsize=figsize)
    
    # January

    # plot trend line
    z = np.polyfit(monthly_data[JANUARY][YEAR], monthly_data[JANUARY][SUNSHINE_HR], deg=1)
    p = np.poly1d(z)
    axes[0, 0].plot(monthly_data[JANUARY][YEAR], p(monthly_data[JANUARY][YEAR]), 'r--',
                    label='Annual Trend')

    # plot monthly annual data and 3-year rolling average
    sns.lineplot(ax=axes[0, 0], data=monthly_data[JANUARY], x=YEAR, y=SUNSHINE_HR, label='Monthly Mean')
    sns.lineplot(ax=axes[0, 0], data=monthly_data[JANUARY], x=YEAR, y=THREE_YEAR_ROLLING_AVG,
                color='black', linestyle='--', label='3-Year Rolling Average')
    axes[0, 0].text(0.12, 0.90, JANUARY, horizontalalignment='center', verticalalignment='center', 
                    transform=axes[0, 0].transAxes, fontsize=17)
    axes[0, 0].set_ylabel(y_label, fontsize=18)
    axes[0, 0].set(ylim=(0, y_max))
    axes[0, 0].set_xlabel('')

    # February
    z = np.polyfit(monthly_data[FEBRUARY][YEAR], monthly_data[FEBRUARY][SUNSHINE_HR], deg=1)
    p = np.poly1d(z)
    axes[0, 1].plot(monthly_data[FEBRUARY][YEAR], p(monthly_data[JANUARY][YEAR]), 'r--')

    sns.lineplot(ax=axes[0, 1], data=monthly_data[FEBRUARY], x=YEAR, y=SUNSHINE_HR)
    sns.lineplot(ax=axes[0, 1], data=monthly_data[FEBRUARY], x=YEAR, y=THREE_YEAR_ROLLING_AVG,
                color='black', linestyle='--')
    axes[0, 1].text(0.12, 0.90, FEBRUARY, horizontalalignment='center', verticalalignment='center', 
                    transform=axes[0, 1].transAxes, fontsize=17)
    axes[0, 1].set_ylabel('')
    axes[0, 1].set(ylim=(0, y_max))
    axes[0, 1].set_xlabel('')


    # March
    z = np.polyfit(monthly_data[MARCH][YEAR], monthly_data[MARCH][SUNSHINE_HR], deg=1)
    p = np.poly1d(z)
    axes[0, 2].plot(monthly_data[MARCH][YEAR], p(monthly_data[MARCH][YEAR]), 'r--')

    sns.lineplot(ax=axes[0, 2], data=monthly_data[MARCH], x=YEAR, y=SUNSHINE_HR)
    sns.lineplot(ax=axes[0, 2], data=monthly_data[MARCH], x=YEAR, y=THREE_YEAR_ROLLING_AVG,
                color='black', linestyle='--')
    axes[0, 2].text(0.12, 0.90, MARCH, horizontalalignment='center', verticalalignment='center', 
                    transform=axes[0, 2].transAxes, fontsize=17)
    axes[0, 2].set_ylabel('')
    axes[0, 2].set(ylim=(0, y_max))
    axes[0, 2].set_xlabel('')


    # April
    z = np.polyfit(monthly_data[APRIL][YEAR], monthly_data[APRIL][SUNSHINE_HR], deg=1)
    p = np.poly1d(z)
    axes[1, 0].plot(monthly_data[APRIL][YEAR], p(monthly_data[APRIL][YEAR]), 'r--')

    sns.lineplot(ax=axes[1, 0], data=monthly_data[APRIL], x=YEAR, y=SUNSHINE_HR)
    sns.lineplot(ax=axes[1, 0], data=monthly_data[APRIL], x=YEAR, y=THREE_YEAR_ROLLING_AVG,
                color='black', linestyle='--')
    axes[1, 0].text(0.12, 0.90, APRIL, horizontalalignment='center', verticalalignment='center', 
                    transform=axes[1, 0].transAxes, fontsize=17)
    axes[1, 0].set_ylabel(y_label, fontsize=18)
    axes[1, 0].set(ylim=(0, y_max))
    axes[1, 0].set_xlabel('')

    # May
    z = np.polyfit(monthly_data[MAY][YEAR], monthly_data[MAY][SUNSHINE_HR], deg=1)
    p = np.poly1d(z)
    axes[1, 1].plot(monthly_data[MAY][YEAR], p(monthly_data[MAY][YEAR]), 'r--')

    sns.lineplot(ax=axes[1, 1], data=monthly_data[MAY], x=YEAR, y=SUNSHINE_HR)
    sns.lineplot(ax=axes[1, 1], data=monthly_data[MAY], x=YEAR, y=THREE_YEAR_ROLLING_AVG,
                color='black', linestyle='--')
    axes[1, 1].text(0.12, 0.90, MAY, horizontalalignment='center', verticalalignment='center', 
                    transform=axes[1, 1].transAxes, fontsize=17)
    axes[1, 1].set_ylabel('')
    axes[1, 1].set(ylim=(0, y_max))
    axes[1, 1].set_xlabel('')


    # June
    z = np.polyfit(monthly_data[JUNE][YEAR], monthly_data[JUNE][SUNSHINE_HR], deg=1)
    p = np.poly1d(z)
    axes[1, 2].plot(monthly_data[JUNE][YEAR], p(monthly_data[JUNE][YEAR]), 'r--',
                    label='Annual Trend')

    sns.lineplot(ax=axes[1, 2], data=monthly_data[JUNE], x=YEAR, y=SUNSHINE_HR, label='Monthly Mean')
    sns.lineplot(ax=axes[1, 2], data=monthly_data[JUNE], x=YEAR, y=THREE_YEAR_ROLLING_AVG,
                color='black', linestyle='--', label='3-Year Rolling Average')
    axes[1, 2].text(0.12, 0.95, JUNE, horizontalalignment='center', verticalalignment='center', 
                    transform=axes[1, 2].transAxes, fontsize=17)
    axes[1, 2].set_ylabel('')
    axes[1, 2].set(ylim=(0, y_max))
    axes[1, 2].set_xlabel('')

    # July
    z = np.polyfit(monthly_data[JULY][YEAR], monthly_data[JULY][SUNSHINE_HR], deg=1)
    p = np.poly1d(z)
    axes[2, 0].plot(monthly_data[JULY][YEAR], p(monthly_data[JULY][YEAR]), 'r--')

    sns.lineplot(ax=axes[2, 0], data=monthly_data[JULY], x=YEAR, y=SUNSHINE_HR)
    sns.lineplot(ax=axes[2, 0], data=monthly_data[JULY], x=YEAR, y=THREE_YEAR_ROLLING_AVG,
                color='black', linestyle='--')

    axes[2, 0].text(0.12, 0.95, JULY, horizontalalignment='center', verticalalignment='center', 
                    transform=axes[2, 0].transAxes, fontsize=17)
    axes[2, 0].set_ylabel(y_label, fontsize=18)
    axes[2, 0].set(ylim=(0, y_max))
    axes[2, 0].set_xlabel('')

    # August
    z = np.polyfit(monthly_data[AUGUST][YEAR], monthly_data[AUGUST][SUNSHINE_HR], deg=1)
    p = np.poly1d(z)
    axes[2, 1].plot(monthly_data[AUGUST][YEAR], p(monthly_data[AUGUST][YEAR]), 'r--')

    sns.lineplot(ax=axes[2, 1], data=monthly_data[AUGUST], x=YEAR, y=SUNSHINE_HR)
    sns.lineplot(ax=axes[2, 1], data=monthly_data[AUGUST], x=YEAR, y=THREE_YEAR_ROLLING_AVG,
                color='black', linestyle='--')

    axes[2, 1].text(0.12, 0.90, AUGUST, horizontalalignment='center', verticalalignment='center', 
                    transform=axes[2, 1].transAxes, fontsize=17)
    axes[2, 1].set_ylabel('')
    axes[2, 1].set(ylim=(0, y_max))
    axes[2, 1].set_xlabel('')

    # September
    z = np.polyfit(monthly_data[SEPTEMBER][YEAR], monthly_data[SEPTEMBER][SUNSHINE_HR], deg=1)
    p = np.poly1d(z)
    axes[2, 2].plot(monthly_data[SEPTEMBER][YEAR], p(monthly_data[SEPTEMBER][YEAR]), 'r--')

    sns.lineplot(ax=axes[2, 2], data=monthly_data[SEPTEMBER], x=YEAR, y=SUNSHINE_HR)
    sns.lineplot(ax=axes[2, 2], data=monthly_data[SEPTEMBER], x=YEAR, y=THREE_YEAR_ROLLING_AVG,
                color='black', linestyle='--')
    axes[2, 2].text(0.15, 0.90, SEPTEMBER, horizontalalignment='center', verticalalignment='center', 
                    transform=axes[2, 2].transAxes, fontsize=17)
    axes[2, 2].set_ylabel('')
    axes[2, 2].set(ylim=(0, y_max))
    axes[2, 2].set_xlabel('')

    # October
    z = np.polyfit(monthly_data[OCTOBER][YEAR], monthly_data[OCTOBER][SUNSHINE_HR], deg=1)
    p = np.poly1d(z)
    axes[3, 0].plot(monthly_data[OCTOBER][YEAR], p(monthly_data[OCTOBER][YEAR]), 'r--',
                    label='Annual Trend')

    sns.lineplot(ax=axes[3, 0], data=monthly_data[OCTOBER], x=YEAR, y=SUNSHINE_HR, label='Monthly Mean')
    sns.lineplot(ax=axes[3, 0], data=monthly_data[OCTOBER], x=YEAR, y=THREE_YEAR_ROLLING_AVG,
                color='black', linestyle='--', label='3-Year Rolling Average')
    axes[3, 0].text(0.12, 0.90, OCTOBER, horizontalalignment='center', verticalalignment='center', 
                    transform=axes[3, 0].transAxes, fontsize=17)
    axes[3, 0].set_ylabel(y_label, fontsize=18)
    axes[3, 0].set(ylim=(0, y_max))
    axes[3, 0].set_xlabel('')

    # November
    z = np.polyfit(monthly_data[NOVEMBER][YEAR], monthly_data[NOVEMBER][SUNSHINE_HR], deg=1)
    p = np.poly1d(z)
    axes[3, 1].plot(monthly_data[NOVEMBER][YEAR], p(monthly_data[NOVEMBER][YEAR]), 'r--')

    sns.lineplot(ax=axes[3, 1], data=monthly_data[NOVEMBER], x=YEAR, y=SUNSHINE_HR)
    sns.lineplot(ax=axes[3, 1], data=monthly_data[NOVEMBER], x=YEAR, y=THREE_YEAR_ROLLING_AVG,
                color='black', linestyle='--')
    axes[3, 1].text(0.12, 0.90, NOVEMBER, horizontalalignment='center', verticalalignment='center', 
                    transform=axes[3, 1].transAxes, fontsize=17)
    axes[3, 1].set_ylabel('')
    axes[3, 1].set(ylim=(0, y_max))
    axes[3, 1].set_xlabel('')

    # December
    z = np.polyfit(monthly_data[DECEMBER][YEAR], monthly_data[DECEMBER][SUNSHINE_HR], deg=1)
    p = np.poly1d(z)
    axes[3, 2].plot(monthly_data[DECEMBER][YEAR], p(monthly_data[DECEMBER][YEAR]), 'r--')

    sns.lineplot(ax=axes[3, 2], data=monthly_data[DECEMBER], x=YEAR, y=SUNSHINE_HR)
    sns.lineplot(ax=axes[3, 2], data=monthly_data[DECEMBER], x=YEAR, y=THREE_YEAR_ROLLING_AVG,
                color='black', linestyle='--')
    axes[3, 2].text(0.12, 0.90, DECEMBER, horizontalalignment='center', verticalalignment='center', 
                    transform=axes[3, 2].transAxes, fontsize=17)
    axes[3, 2].set_ylabel('')
    axes[3, 2].set(ylim=(0, y_max))
    axes[3, 2].set_xlabel('')

    fig.tight_layout()

    if name:
        plt.savefig(f'{name}.png')


