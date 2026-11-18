"""
[Kaggle 1단계] 넓은 탐색 (Broad search)

- 목적: 노이즈가 있는 이 데이터에 어떤 모델 계열이 잘 맞는지 대략적으로 확인
- 검증: Stratified 5-fold CV (모든 모델이 같은 fold를 사용 -> 공정한 비교)
- 순서: 0) baseline 기록
        1) Decision Tree (단일 모델)
        2) Bagging / Random Forest
        3) AdaBoost
        4) HistGradientBoosting
- 결과: results/stage1/ 폴더
        baseline.txt          : baseline 성능 (notebook 방식 holdout + 5-fold CV)
        submission_baseline.csv : baseline 제출 파일 (LB와 CV 비교용)
        results.csv           : 모든 조합의 CV 결과 (조합 하나 끝날 때마다 저장)
        best_per_model.csv    : 모델 계열별 최고 조합 -> 2단계 후보
        best_setting.txt      : 1단계 전체 최고 설정 (SELECT_METRIC 기준)
        submission_stage1_best.csv : 최고 설정을 전체 train으로 다시 학습한 제출 파일
- 중간에 멈춰도 다시 실행하면 이미 끝난 조합은 건너뛰고 이어서 진행

실행:
    cd Code/kaggle
    python -u stage1_broad_search.py | tee results_stage1_log.txt
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / 'results' / 'stage1'

RANDOM_STATE = 42   # CV fold 분할과 모든 모델에 같은 seed 사용 (재현성)
N_FOLDS = 5
CV_JOBS = 5         # 5개 fold를 동시에 학습
RF_JOBS = 4         # Random Forest 하나가 쓰는 CPU 수 (5 fold x 4 = 20 core)

# ==================== EDIT HERE: START ====================
# 최고 설정을 고르는 기준: Kaggle 평가 지표에 맞춰 선택
#   'cv_acc_mean' (accuracy) 또는 'cv_f1_mean' (macro F1)
SELECT_METRIC = 'cv_acc_mean'

# 넓은 탐색이므로 값을 듬성듬성(대략 로그 간격) 잡음

# 1) Decision Tree
DT_MAX_DEPTH = [4, 8, 12, 16, 24, None]
DT_MIN_SAMPLES_LEAF = [1, 5, 20, 50, 100]
DT_CRITERION = ['gini', 'entropy']
DT_CLASS_WEIGHT = [None, 'balanced']

# 2) Random Forest
#    max_features=1.0 이면 모든 feature를 쓰므로 Bagging과 같음 -> 'Bagging'으로 기록
#    트리 개수는 늘려도 과적합되지 않으므로(평균만 안정됨) 고정
RF_N_ESTIMATORS = 300
RF_MAX_FEATURES = ['sqrt', 0.2, 0.4, 0.7, 1.0]
RF_MIN_SAMPLES_LEAF = [1, 2, 5, 10, 20, 50]
RF_CLASS_WEIGHT = [None, 'balanced']

# 3) AdaBoost (base model = 깊이 max_depth인 Decision Tree)
ADA_MAX_DEPTH = [1, 3, 6, 10]
ADA_N_ESTIMATORS = [50, 200, 500]
ADA_LEARNING_RATE = [0.1, 0.5, 1.0]

# 4) HistGradientBoosting
#    트리 개수는 early stopping이 정함 (학습 데이터의 10%로 검증, 20번 연속 개선 없으면 멈춤)
HGB_LEARNING_RATE = [0.03, 0.1, 0.3]
HGB_MAX_LEAF_NODES = [15, 31, 63]
HGB_MIN_SAMPLES_LEAF = [20, 100]
HGB_L2 = [0.0, 1.0]
HGB_CLASS_WEIGHT = [None, 'balanced']
# ==================== EDIT HERE: END ======================


def load_kaggle_data():
    train = pd.read_csv(ROOT / 'train_data.csv')
    test = pd.read_csv(ROOT / 'test_data.csv')
    X = train.drop(columns=['id', 'Cover_Type'])
    y = train['Cover_Type']
    X_test = test.drop(columns='id')
    return X, y, X_test, test['id']


def params_to_text(params):
    return ', '.join(f'{k}={v}' for k, v in params.items())


def load_results():
    """이전에 저장된 결과가 있으면 불러옴 (이어서 실행용)."""
    result_csv = RESULT_DIR / 'results.csv'
    if result_csv.exists():
        return pd.read_csv(result_csv).to_dict('records')
    return []


def run_cv(model_name, model, params, X, y, cv, results, progress=''):
    """모델 하나를 5-fold CV로 평가하고 results에 추가 + 저장."""
    params_text = params_to_text(params)

    # 이미 끝난 조합은 건너뜀
    for row in results:
        if row['model'] == model_name and row['params'] == params_text:
            return

    start = time.perf_counter()
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring={'acc': 'accuracy', 'f1': 'f1_macro'},
        return_train_score=True,
        n_jobs=CV_JOBS,
    )
    elapsed = time.perf_counter() - start

    row = {
        'model': model_name,
        'params': params_text,
        'cv_acc_mean': scores['test_acc'].mean(),
        'cv_acc_std': scores['test_acc'].std(),
        'cv_f1_mean': scores['test_f1'].mean(),
        'cv_f1_std': scores['test_f1'].std(),
        'train_acc_mean': scores['train_acc'].mean(),
        'gap': scores['train_acc'].mean() - scores['test_acc'].mean(),  # 클수록 노이즈를 외움
        'time_sec': elapsed,
    }
    row.update(params)  # 파라미터별 열 (나중에 그래프 그릴 때 사용)
    results.append(row)
    pd.DataFrame(results).to_csv(RESULT_DIR / 'results.csv', index=False)

    print(f'{progress}[{model_name}] {params_text}\n'
          f'    CV acc {row["cv_acc_mean"]:.4f} ± {row["cv_acc_std"]:.4f} | '
          f'CV F1 {row["cv_f1_mean"]:.4f} | train acc {row["train_acc_mean"]:.4f} | '
          f'gap {row["gap"]:.4f} | {elapsed:.1f}s', flush=True)


def record_baseline(X, y, X_test, test_id, cv, results):
    """제공된 baseline(기본값 Decision Tree)의 성능을 먼저 기록."""
    print('========== 0) Baseline ==========', flush=True)

    # (a) baseline notebook과 똑같은 방식: 75/25 holdout 한 번
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    holdout_acc = accuracy_score(y_valid, model.predict(X_valid))

    # (b) 이번 탐색과 같은 5-fold CV
    run_cv('Baseline', DecisionTreeClassifier(random_state=42), {'default': True},
           X, y, cv, results)
    base = [row for row in results if row['model'] == 'Baseline'][0]

    # (c) baseline 제출 파일 (전체 train으로 학습) -> LB 점수와 CV 점수 비교용
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    submission = pd.DataFrame({'id': test_id, 'Cover_Type': model.predict(X_test)})
    submission.to_csv(RESULT_DIR / 'submission_baseline.csv', index=False)

    text = (
        'Baseline: DecisionTreeClassifier(random_state=42), 기본값\n'
        f'(a) holdout 75/25 accuracy (notebook 방식): {holdout_acc:.4f}\n'
        f'(b) 5-fold CV accuracy: {base["cv_acc_mean"]:.4f} ± {base["cv_acc_std"]:.4f}\n'
        f'    5-fold CV macro F1: {base["cv_f1_mean"]:.4f} ± {base["cv_f1_std"]:.4f}\n'
        f'    train accuracy: {base["train_acc_mean"]:.4f} (gap {base["gap"]:.4f})\n'
        '(c) Kaggle public LB: (제출 후 직접 기록)\n'
    )
    (RESULT_DIR / 'baseline.txt').write_text(text, encoding='utf-8')
    print(text, flush=True)


def print_summary(results):
    df = pd.DataFrame(results).sort_values(SELECT_METRIC, ascending=False)
    cols = ['model', 'params', 'cv_acc_mean', 'cv_acc_std', 'cv_f1_mean', 'train_acc_mean', 'gap']

    best = df.groupby('model').head(1)
    best.to_csv(RESULT_DIR / 'best_per_model.csv', index=False)

    print(f'\n========== 모델 계열별 최고 조합 ({SELECT_METRIC} 기준) ==========')
    print(best[cols].to_string(index=False, float_format='%.4f'))
    print('\n========== 전체 상위 10개 ==========')
    print(df[cols].head(10).to_string(index=False, float_format='%.4f'))


def make_candidates():
    """1단계에서 평가할 (모델 이름, 파라미터, 모델) 목록을 만듦."""
    candidates = []

    # 1) Decision Tree (단일 모델)
    for max_depth in DT_MAX_DEPTH:
        for min_samples_leaf in DT_MIN_SAMPLES_LEAF:
            for criterion in DT_CRITERION:
                for class_weight in DT_CLASS_WEIGHT:
                    params = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf,
                              'criterion': criterion, 'class_weight': class_weight}
                    model = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
                    candidates.append(('DecisionTree', params, model))

    # 2) Bagging / Random Forest
    for max_features in RF_MAX_FEATURES:
        for min_samples_leaf in RF_MIN_SAMPLES_LEAF:
            for class_weight in RF_CLASS_WEIGHT:
                params = {'max_features': max_features, 'min_samples_leaf': min_samples_leaf,
                          'class_weight': class_weight}
                model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, n_jobs=RF_JOBS,
                                               random_state=RANDOM_STATE, **params)
                model_name = 'Bagging' if max_features == 1.0 else 'RandomForest'
                candidates.append((model_name, params, model))

    # 3) AdaBoost
    for max_depth in ADA_MAX_DEPTH:
        for n_estimators in ADA_N_ESTIMATORS:
            for learning_rate in ADA_LEARNING_RATE:
                params = {'max_depth': max_depth, 'n_estimators': n_estimators,
                          'learning_rate': learning_rate}
                model = AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE),
                    n_estimators=n_estimators, learning_rate=learning_rate,
                    random_state=RANDOM_STATE,
                )
                candidates.append(('AdaBoost', params, model))

    # 4) HistGradientBoosting
    for learning_rate in HGB_LEARNING_RATE:
        for max_leaf_nodes in HGB_MAX_LEAF_NODES:
            for min_samples_leaf in HGB_MIN_SAMPLES_LEAF:
                for l2 in HGB_L2:
                    for class_weight in HGB_CLASS_WEIGHT:
                        params = {'learning_rate': learning_rate, 'max_leaf_nodes': max_leaf_nodes,
                                  'min_samples_leaf': min_samples_leaf, 'l2_regularization': l2,
                                  'class_weight': class_weight}
                        model = HistGradientBoostingClassifier(
                            max_iter=1000, early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=20, random_state=RANDOM_STATE, **params,
                        )
                        candidates.append(('HistGradientBoosting', params, model))

    return candidates


def make_best_submission(results, candidates, X, y, X_test, test_id):
    """SELECT_METRIC이 가장 높은 설정을 전체 train으로 다시 학습해서 제출 파일 생성."""
    # 이번 탐색 목록(candidates)에 있는 조합 중에서 최고를 찾음 (Baseline 제외)
    best_row, best_name, best_params, best_model = None, None, None, None
    for model_name, params, model in candidates:
        for row in results:
            if row['model'] == model_name and row['params'] == params_to_text(params):
                if best_row is None or row[SELECT_METRIC] > best_row[SELECT_METRIC]:
                    best_row, best_name, best_params, best_model = row, model_name, params, model

    print(f'\n========== 1단계 최고 설정으로 제출 파일 생성 ({SELECT_METRIC} 기준) ==========')
    print(f'[{best_name}] {params_to_text(best_params)}', flush=True)

    # 5-fold CV는 train의 80%로 학습했지만, 제출용은 train 전체(100%)로 다시 학습
    best_model.fit(X, y)
    submission = pd.DataFrame({'id': test_id, 'Cover_Type': best_model.predict(X_test)})
    submission.to_csv(RESULT_DIR / 'submission_stage1_best.csv', index=False)

    text = (
        f'Stage 1 best ({SELECT_METRIC} 기준)\n'
        f'model: {best_name}\n'
        f'params: {params_to_text(best_params)}\n'
        f'5-fold CV accuracy: {best_row["cv_acc_mean"]:.4f} ± {best_row["cv_acc_std"]:.4f}\n'
        f'5-fold CV macro F1: {best_row["cv_f1_mean"]:.4f} ± {best_row["cv_f1_std"]:.4f}\n'
        f'train accuracy: {best_row["train_acc_mean"]:.4f} (gap {best_row["gap"]:.4f})\n'
        'Kaggle public LB: (제출 후 직접 기록)\n'
    )
    (RESULT_DIR / 'best_setting.txt').write_text(text, encoding='utf-8')
    print(text, flush=True)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, X_test, test_id = load_kaggle_data()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = load_results()
    total_start = time.perf_counter()

    # 0) Baseline 먼저 기록
    record_baseline(X, y, X_test, test_id, cv, results)

    # 1)~4) 넓은 탐색: Decision Tree -> Bagging / RF -> AdaBoost -> HistGradientBoosting
    candidates = make_candidates()
    print(f'========== 1)~4) 넓은 탐색: 총 {len(candidates)}개 조합 ==========', flush=True)
    for i, (model_name, params, model) in enumerate(candidates):
        run_cv(model_name, model, params, X, y, cv, results,
               progress=f'({i + 1}/{len(candidates)}) ')

    print_summary(results)

    # 5) 1단계 최고 설정으로 Kaggle 제출 파일 생성
    make_best_submission(results, candidates, X, y, X_test, test_id)

    print(f'Total time: {(time.perf_counter() - total_start) / 60:.1f} min')
    print(f'Results: {RESULT_DIR}')


if __name__ == '__main__':
    main()
