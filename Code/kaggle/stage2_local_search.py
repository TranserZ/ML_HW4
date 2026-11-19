"""
[Kaggle 2단계] 국소 탐색 (Local search)

- 목적: 1단계 상위 계열(Random Forest, HistGradientBoosting)의 최적값 주변을 촘촘하게 탐색
- 검증: 1단계와 같은 Stratified 5-fold CV (seed 42) -> 1단계 점수와 그대로 비교 가능
- 1단계 결과(results/stage1/results.csv)로 정한 범위
    Random Forest : min_samples_leaf=2가 모든 max_features에서 최고 (1, 5는 더 낮음) -> 1~4 촘촘히
                    max_features는 sqrt~0.7에서 거의 차이 없음 -> 더 작은 쪽(0.1)까지 확인
                    + 노이즈 대응용 max_samples(트리마다 쓰는 데이터 비율), criterion 추가
    HistGB        : 최고값이 탐색 범위 끝에 있었음 (max_leaf_nodes=63은 최대값, min_samples_leaf=20은 최소값)
                    -> 범위를 바깥쪽으로 넓혀서 확인 + max_features(트리마다 쓰는 feature 비율) 추가
    제외          : class_weight='balanced' (accuracy, macro F1 모두 낮아짐)
                    AdaBoost (최고 0.550으로 RF보다 0.028 낮고, 가장 오래 걸림)
                    Decision Tree (단일 모델 비교는 1단계 결과 사용)
- 순서: 1) Random Forest 격자
        2) 최고 RF 설정으로 트리 개수만 늘려 보기
        3) HistGradientBoosting 격자
        4) 최고 설정으로 Kaggle 제출 파일 생성
- 결과: results/stage2/ 폴더
        results.csv           : 모든 조합의 CV 결과 (조합 하나 끝날 때마다 저장)
        best_per_model.csv    : 모델 계열별 최고 조합 -> 3단계 Voting/Stacking 재료
        close_to_best.csv     : 최고 점수와 1 std 이내 조합 (사실상 동점) -> 3단계 반복 CV 후보
        best_setting.txt      : 2단계 전체 최고 설정
        submission_stage2_best.csv : 최고 설정을 전체 train으로 다시 학습한 제출 파일
- 중간에 멈춰도 다시 실행하면 이미 끝난 조합은 건너뛰고 이어서 진행

실행:
    cd Code/kaggle
    python -u stage2_local_search.py | tee results_stage2_log.txt
"""
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / 'results' / 'stage2'
STAGE1_CSV = ROOT / 'results' / 'stage1' / 'results.csv'

RANDOM_STATE = 42   # 1단계와 같은 seed -> 같은 fold
N_FOLDS = 5
CV_JOBS = 5         # 5개 fold를 동시에 학습
RF_JOBS = 4         # Random Forest 하나가 쓰는 CPU 수 (5 fold x 4 = 20 core)

# ==================== EDIT HERE: START ====================
# 최고 설정을 고르는 기준: 'cv_acc_mean' (accuracy) 또는 'cv_f1_mean' (macro F1)
SELECT_METRIC = 'cv_acc_mean'

# 1) Random Forest (1단계 최고: max_features=0.2, min_samples_leaf=2 -> 0.5786)
RF_N_ESTIMATORS = 300
RF_MAX_FEATURES = [0.1, 'sqrt', 0.2, 0.3, 0.4, 0.5]   # sqrt = 54개 중 약 7개 (13%)
RF_MIN_SAMPLES_LEAF = [1, 2, 3, 4]
RF_MAX_SAMPLES = [None, 0.8, 0.6]     # 트리마다 뽑는 bootstrap 크기 (None = N개 전부)
RF_CRITERION = ['gini', 'entropy']

# 2) 최고 RF 설정으로 트리 개수만 늘려 보기 (300은 1)에서 이미 평가)
RF_N_ESTIMATORS_CHECK = [600, 1000]

# 3) HistGradientBoosting (1단계 최고: lr=0.1, max_leaf_nodes=63, min_samples_leaf=20 -> 0.5597)
HGB_LEARNING_RATE = [0.03, 0.06, 0.1]
HGB_MAX_LEAF_NODES = [63, 127, 255]
HGB_MIN_SAMPLES_LEAF = [5, 10, 20, 40]
HGB_L2 = [0.0, 1.0, 10.0]
HGB_MAX_FEATURES = [1.0, 0.5]         # 트리 split마다 쓰는 feature 비율 (1.0 = 전부)
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


def run_all(candidates, X, y, cv, results):
    for i, (model_name, params, model) in enumerate(candidates):
        run_cv(model_name, model, params, X, y, cv, results,
               progress=f'({i + 1}/{len(candidates)}) ')


def find_best(results, candidates):
    """candidates 중에서 SELECT_METRIC이 가장 높은 (결과, 모델 이름, 파라미터, 모델)을 찾음."""
    best = None
    for model_name, params, model in candidates:
        for row in results:
            if row['model'] == model_name and row['params'] == params_to_text(params):
                if best is None or row[SELECT_METRIC] > best[0][SELECT_METRIC]:
                    best = (row, model_name, params, model)
    return best


def make_rf_candidates():
    candidates = []
    for max_features in RF_MAX_FEATURES:
        for min_samples_leaf in RF_MIN_SAMPLES_LEAF:
            for max_samples in RF_MAX_SAMPLES:
                for criterion in RF_CRITERION:
                    params = {'n_estimators': RF_N_ESTIMATORS, 'max_features': max_features,
                              'min_samples_leaf': min_samples_leaf, 'max_samples': max_samples,
                              'criterion': criterion}
                    model = RandomForestClassifier(n_jobs=RF_JOBS, random_state=RANDOM_STATE, **params)
                    candidates.append(('RandomForest', params, model))
    return candidates


def make_n_estimators_candidates(best_rf_params):
    """최고 RF 설정에서 트리 개수만 바꾼 후보."""
    candidates = []
    for n_estimators in RF_N_ESTIMATORS_CHECK:
        params = dict(best_rf_params)
        params['n_estimators'] = n_estimators
        model = RandomForestClassifier(n_jobs=RF_JOBS, random_state=RANDOM_STATE, **params)
        candidates.append(('RandomForest', params, model))
    return candidates


def make_hgb_candidates():
    candidates = []
    for learning_rate in HGB_LEARNING_RATE:
        for max_leaf_nodes in HGB_MAX_LEAF_NODES:
            for min_samples_leaf in HGB_MIN_SAMPLES_LEAF:
                for l2 in HGB_L2:
                    for max_features in HGB_MAX_FEATURES:
                        params = {'learning_rate': learning_rate, 'max_leaf_nodes': max_leaf_nodes,
                                  'min_samples_leaf': min_samples_leaf, 'l2_regularization': l2,
                                  'max_features': max_features}
                        model = HistGradientBoostingClassifier(
                            max_iter=1000, early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=20, random_state=RANDOM_STATE, **params,
                        )
                        candidates.append(('HistGradientBoosting', params, model))
    return candidates


def print_summary(results):
    df = pd.DataFrame(results).sort_values(SELECT_METRIC, ascending=False)
    std_col = SELECT_METRIC.replace('mean', 'std')
    cols = ['model', 'params', 'cv_acc_mean', 'cv_acc_std', 'cv_f1_mean', 'train_acc_mean', 'gap']

    best = df.groupby('model').head(1)
    best.to_csv(RESULT_DIR / 'best_per_model.csv', index=False)
    print(f'\n========== 모델 계열별 최고 조합 ({SELECT_METRIC} 기준) ==========')
    print(best[cols].to_string(index=False, float_format='%.4f'))

    # 최고 점수와 차이가 1 std보다 작은 조합 = 이번 CV로는 우열을 가리기 어려움 (사실상 동점)
    top = df.iloc[0]
    close = df[df[SELECT_METRIC] >= top[SELECT_METRIC] - top[std_col]]
    close.to_csv(RESULT_DIR / 'close_to_best.csv', index=False)
    print(f'\n========== 최고 점수와 1 std 이내 조합: {len(close)}개 (상위 15개 표시) ==========')
    print(close[cols].head(15).to_string(index=False, float_format='%.4f'))

    # 1단계 최고 점수와 비교
    if STAGE1_CSV.exists():
        stage1 = pd.read_csv(STAGE1_CSV)
        stage1_best = stage1[stage1['model'] != 'Baseline'][SELECT_METRIC].max()
        print(f'\n1단계 최고 {SELECT_METRIC}: {stage1_best:.4f} -> 2단계 최고: {top[SELECT_METRIC]:.4f} '
              f'({top[SELECT_METRIC] - stage1_best:+.4f})')


def make_best_submission(best, X, y, X_test, test_id):
    """최고 설정을 전체 train으로 다시 학습해서 제출 파일 생성."""
    best_row, best_name, best_params, best_model = best

    print(f'\n========== 2단계 최고 설정으로 제출 파일 생성 ({SELECT_METRIC} 기준) ==========')
    print(f'[{best_name}] {params_to_text(best_params)}', flush=True)

    # 5-fold CV는 train의 80%로 학습했지만, 제출용은 train 전체(100%)로 다시 학습
    best_model.fit(X, y)
    submission = pd.DataFrame({'id': test_id, 'Cover_Type': best_model.predict(X_test)})
    submission.to_csv(RESULT_DIR / 'submission_stage2_best.csv', index=False)

    text = (
        f'Stage 2 best ({SELECT_METRIC} 기준)\n'
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

    # 1) Random Forest 격자
    rf_candidates = make_rf_candidates()
    print(f'========== 1) Random Forest: {len(rf_candidates)}개 조합 ==========', flush=True)
    run_all(rf_candidates, X, y, cv, results)

    # 2) 최고 RF 설정으로 트리 개수만 늘려 보기
    best_rf_params = find_best(results, rf_candidates)[2]
    tree_candidates = make_n_estimators_candidates(best_rf_params)
    print(f'========== 2) 트리 개수 확인: {RF_N_ESTIMATORS_CHECK} ==========', flush=True)
    run_all(tree_candidates, X, y, cv, results)

    # 3) HistGradientBoosting 격자
    hgb_candidates = make_hgb_candidates()
    print(f'========== 3) HistGradientBoosting: {len(hgb_candidates)}개 조합 ==========', flush=True)
    run_all(hgb_candidates, X, y, cv, results)

    print_summary(results)

    # 4) 2단계 최고 설정으로 Kaggle 제출 파일 생성
    best = find_best(results, rf_candidates + tree_candidates + hgb_candidates)
    make_best_submission(best, X, y, X_test, test_id)

    print(f'Total time: {(time.perf_counter() - total_start) / 60:.1f} min')
    print(f'Results: {RESULT_DIR}')


if __name__ == '__main__':
    main()
