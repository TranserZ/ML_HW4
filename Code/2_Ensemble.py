from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from models.Ensemble import Bagging, AdaBoost
from ensemble_utils import load_data, save_figures


"""
Run the Bagging / AdaBoost experiments.

Complete EDIT HERE 1-5 in models/Ensemble.py.
In this file, change ONLY the settings inside EDIT HERE, then run:
    python 2_Ensemble.py
"""

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'ensemble'
RESULT_DIR = ROOT / 'results' / 'ensemble'

RANDOM_STATE = 42  # Fixed for the assignment.

# ==================== EDIT HERE: START ====================
NUM_TREES = [1, 10, 50]
# ==================== EDIT HERE: END ======================


def evaluate(model, X_train, y_train, X_test, y_test):
    """Train one model and return its training/test accuracies."""
    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    train_accuracy = np.mean(train_prediction == y_train)
    test_accuracy = np.mean(test_prediction == y_test)
    return train_accuracy, test_accuracy


def run_experiment(X_train, y_train, X_test, y_test):
    """Compare single trees, Bagging, and AdaBoost on one training set."""
    models = [
        DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
        DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE),
    ]
    model_names = ['Single shallow tree', 'Single deep tree']
    tree_counts = [1, 1]

    for num_trees in NUM_TREES:
        models.append(Bagging(n_estimators=num_trees, random_state=RANDOM_STATE))
        model_names.append('Bagging')
        tree_counts.append(num_trees)

        models.append(AdaBoost(n_estimators=num_trees, random_state=RANDOM_STATE))
        model_names.append('AdaBoost')
        tree_counts.append(num_trees)

    results = []
    fitted_models = {}

    for i in range(len(models)):
        model = models[i]
        model_name = model_names[i]
        num_trees = tree_counts[i]

        train_accuracy, test_accuracy = evaluate(
            model, X_train, y_train, X_test, y_test,
        )

        results.append({
            'model': model_name,
            'n_estimators': num_trees,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
        })

        # Plot the largest requested ensemble, even when 50 is not selected.
        if model_name in ['Single shallow tree', 'Single deep tree'] or num_trees == max(NUM_TREES):
            fitted_models[model_name] = model

    return results, fitted_models


def main():
    # Provided: check the editable experiment settings.
    if not isinstance(RANDOM_STATE, int) or not 0 <= RANDOM_STATE < 2**32:
        raise ValueError('RANDOM_STATE must be an integer from 0 to 2**32 - 1.')
    if not isinstance(NUM_TREES, list) or not NUM_TREES:
        raise ValueError('NUM_TREES must be a nonempty list, e.g. [1, 10, 50].')
    if any(not isinstance(n, int) or n < 1 for n in NUM_TREES):
        raise ValueError('Each value in NUM_TREES must be a positive integer.')
    if RANDOM_STATE + max(NUM_TREES) - 1 >= 2**32:
        raise ValueError('RANDOM_STATE + max(NUM_TREES) must not exceed 2**32.')
    if len(set(NUM_TREES)) != len(NUM_TREES):
        raise ValueError('Do not repeat values in NUM_TREES.')

    # 1. Load the training and test sets.
    X_train, y_train = load_data(DATA_DIR / 'train.csv')
    X_test, y_test = load_data(DATA_DIR / 'test.csv')

    # 2. Compare the models with the selected tree counts.
    start_time = time.perf_counter()
    all_results, fitted_models = run_experiment(X_train, y_train, X_test, y_test)
    elapsed = time.perf_counter() - start_time

    # 3. Print and save the results for the assignment tables.
    run_dir = RESULT_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(all_results)
    results.to_csv(run_dir / 'metrics.csv', index=False)

    result_text = results.to_string(index=False, float_format='%.4f')
    time_text = f'Fit/evaluation time: {elapsed:.3f} s'
    settings_text = (
        f'RANDOM_STATE = {RANDOM_STATE}\n'
        f'NUM_TREES = {NUM_TREES}\n'
    )
    print(settings_text)
    print(result_text)
    print(time_text)
    (run_dir / 'run_log.txt').write_text(
        settings_text + result_text + '\n' + time_text + '\n', encoding='utf-8',
    )

    # 4. Save figures using the provided helper (no student edits required).
    save_figures(
        X_train, y_train, fitted_models, results, run_dir,
    )
    print(f'Tables and figures: {run_dir}')


if __name__ == '__main__':
    main()
