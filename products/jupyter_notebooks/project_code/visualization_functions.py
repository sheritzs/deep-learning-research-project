
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
