"""Provided data loading and plotting helpers. No student edits required."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(path):
    """Read x1/x2 features and -1/+1 labels from a CSV file."""
    data = pd.read_csv(path)
    X = data[['x1', 'x2']].to_numpy()
    y = data['label'].to_numpy()
    return X, y


def save_figures(X, y, fitted, table, output_dir):
    """Save data, decision boundaries, and accuracy curves."""
    fig, ax = plt.subplots(figsize=(4.5, 3.7), constrained_layout=True)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=15, vmin=-1, vmax=1)
    ax.set(xlabel='x1', ylabel='x2', title='Training data')
    fig.savefig(output_dir / 'dataset.png', dpi=160)
    plt.close(fig)

    gx, gy = np.meshgrid(np.linspace(-1.8, 2.8, 220), np.linspace(-1.3, 1.9, 180))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), constrained_layout=True)
    for ax, name in zip(axes, ['Single shallow tree', 'Single deep tree', 'Bagging', 'AdaBoost']):
        model = fitted[name]
        ax.contourf(gx, gy, model.predict(grid).reshape(gx.shape), levels=[-2, 0, 2],
                    colors=['#bdd7ee', '#f8c9c9'], alpha=0.7)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=7, vmin=-1, vmax=1)
        count = len(model.estimators_) if name in ['Bagging', 'AdaBoost'] else 1
        ax.set(title=f'{name} ({count} trees)', xlabel='x1', ylabel='x2')
    fig.savefig(output_dir / 'decision_boundaries.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.7), constrained_layout=True)
    for name in ['Bagging', 'AdaBoost']:
        sub = table[table.model == name].sort_values('n_estimators')
        ax.plot(sub.n_estimators, sub.test_accuracy, 'o-', label=f'{name} test')
        ax.plot(sub.n_estimators, sub.train_accuracy, 'o--', label=f'{name} train')
    ax.set(title='Ensemble accuracy', xlabel='Number of trees', ylabel='Accuracy', ylim=(0.65, 1.02))
    ax.legend(fontsize=8)
    fig.savefig(output_dir / 'learning_curves.png', dpi=160)
    plt.close(fig)
