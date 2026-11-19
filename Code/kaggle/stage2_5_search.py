"""
[Kaggle 2.5단계] Bagging / AdaBoost 추가 탐색

- 목적: 2단계(RF, HistGB)에서 빠진 Bagging과 AdaBoost도 각자 최고 상태로 만들어서
        "단일 모델 vs 앙상블" 비교를 공정하게 하기
- 검증: 1, 2단계와 같은 Stratified 5-fold CV (seed 42) -> 점수를 그대로 비교 가능
- 1단계 결과(results/stage1/results.csv)로 정한 범위
    Bagging  : RandomForest에서 max_features=1.0 (모든 feature 사용) = 트리 Bagging
               1단계 최고 min_samples_leaf=2 (0.5762) -> 2단계 RF와 같은 범위로 촘촘히
    AdaBoost : 1단계 최고(depth 10, 500개, lr 0.1)가 세 값 모두 범위 끝
               depth가 깊을수록 계속 좋아짐 (1: 0.49, 3: 0.51, 6: 0.525, 10: 0.550)
               -> depth를 더 깊게 + base tree에 min_samples_leaf 규제 추가 + lr를 더 작게
               n_estimators는 50 -> 500 차이가 +0.003뿐이라 300으로 고정
- 순서: 1) Bagging 격자 -> 2) AdaBoost 격자 -> 3) 최고 설정으로 제출 파일 생성
- 결과: results/stage2_5/ 폴더
        results.csv           : 모든 조합의 CV 결과 (조합 하나 끝날 때마다 저장)
        best_per_model.csv    : Bagging, AdaBoost 최고 조합 -> 3단계 재료
        best_setting.txt      : 2.5단계 전체 최고 설정
        submission_stage2_5_best.csv : 최고 설정을 전체 train으로 다시 학습한 제출 파일
- 중간에 멈춰도 다시 실행하면 이미 끝난 조합은 건너뛰고 이어서 진행

실행 (2단계가 끝난 뒤에 실행 권장 - 동시에 돌리면 CPU/메모리를 나눠 써서 둘 다 느려짐):
    cd Code/kaggle
    python -u stage2_5_search.py | tee results_stage2_5_log.txt
"""
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / 'results' / 'stage2_5'
STAGE1_CSV = ROOT / 'results' / 'stage1' / 'results.csv'

RANDOM_STATE = 42   # 1, 2단계와 같은 seed -> 같은 fold
N_FOLDS = 5
CV_JOBS = 5         # 5개 fold를 동시에 학습
RF_JOBS = 4         # Bagging 하나가 쓰는 CPU 수 (5 fold x 4 = 20 core)

# ==================== EDIT HERE: START ====================
# 최고 설정을 고르는 기준: 'cv_acc_mean' (accuracy) 또는 'cv_f1_mean' (macro F1)
SELECT_METRIC = 'cv_acc_mean'

# 1) Bagging (1단계 최고: min_samples_leaf=2 -> 0.5762)
BAG_N_ESTIMATORS = 300
BAG_MIN_SAMPLES_LEAF = [1, 2, 3, 4]
BAG_MAX_SAMPLES = [None, 0.8, 0.6]    # 트리마다 뽑는 bootstrap 크기 (None = N개 전부)
BAG_CRITERION = ['gini', 'entropy']

# 2) AdaBoost (1단계 최고: depth 10, n_estimators 500, lr 0.1 -> 0.5503)
ADA_N_ESTIMATORS = 300
ADA_MAX_DEPTH = [10, 15, 20]           # base tree 깊이
ADA_MIN_SAMPLES_LEAF = [1, 5, 20]      # base tree 잎의 최소 샘플 수 (노이즈 샘플을 따로 외우지 못하게)
ADA_LEARNING_RATE = [0.03, 0.1]
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


def make_bagging_candidates():
    candidates = []
    for min_samples_leaf in BAG_MIN_SAMPLES_LEAF:
        for max_samples in BAG_MAX_SAMPLES:
            for criterion in BAG_CRITERION:
                params = {'n_estimators': BAG_N_ESTIMATORS, 'max_features': 1.0,
                          'min_samples_leaf': min_samples_leaf, 'max_samples': max_samples,
                          'criterion': criterion}
                model = RandomForestClassifier(n_jobs=RF_JOBS, random_state=RANDOM_STATE, **params)
                candidates.append(('Bagging', params, model))
    return candidates


def make_adaboost_candidates():
    candidates = []
    for max_depth in ADA_MAX_DEPTH:
        for min_samples_leaf in ADA_MIN_SAMPLES_LEAF:
            for learning_rate in ADA_LEARNING_RATE:
                params = {'n_estimators': ADA_N_ESTIMATORS, 'max_depth': max_depth,
                          'min_samples_leaf': min_samples_leaf, 'learning_rate': learning_rate}
                model = AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=max_depth,
                                                     min_samples_leaf=min_samples_leaf,
                                                     random_state=RANDOM_STATE),
                    n_estimators=ADA_N_ESTIMATORS, learning_rate=learning_rate,
                    random_state=RANDOM_STATE,
                )
                candidates.append(('AdaBoost', params, model))
    return candidates


def print_summary(results):
    df = pd.DataFrame(results).sort_values(SELECT_METRIC, ascending=False)
    cols = ['model', 'params', 'cv_acc_mean', 'cv_acc_std', 'cv_f1_mean', 'train_acc_mean', 'gap']

    best = df.groupby('model').head(1)
    best.to_csv(RESULT_DIR / 'best_per_model.csv', index=False)
    print(f'\n========== 모델 계열별 최고 조합 ({SELECT_METRIC} 기준) ==========')
    print(best[cols].to_string(index=False, float_format='%.4f'))

    print(f'\n========== 모델별 상위 5개 ==========')
    print(df.groupby('model').head(5)[cols].to_string(index=False, float_format='%.4f'))

    # 1단계 같은 모델의 최고 점수와 비교
    if STAGE1_CSV.exists():
        stage1 = pd.read_csv(STAGE1_CSV)
        print(f'\n========== 1단계 대비 ({SELECT_METRIC}) ==========')
        for _, row in best.iterrows():
            stage1_best = stage1[stage1['model'] == row['model']][SELECT_METRIC].max()
            print(f'{row["model"]}: 1단계 {stage1_best:.4f} -> 2.5단계 {row[SELECT_METRIC]:.4f} '
                  f'({row[SELECT_METRIC] - stage1_best:+.4f})')


def make_best_submission(best, X, y, X_test, test_id):
    """최고 설정을 전체 train으로 다시 학습해서 제출 파일 생성."""
    best_row, best_name, best_params, best_model = best

    print(f'\n========== 2.5단계 최고 설정으로 제출 파일 생성 ({SELECT_METRIC} 기준) ==========')
    print(f'[{best_name}] {params_to_text(best_params)}', flush=True)

    # 5-fold CV는 train의 80%로 학습했지만, 제출용은 train 전체(100%)로 다시 학습
    best_model.fit(X, y)
    submission = pd.DataFrame({'id': test_id, 'Cover_Type': best_model.predict(X_test)})
    submission.to_csv(RESULT_DIR / 'submission_stage2_5_best.csv', index=False)

    text = (
        f'Stage 2.5 best ({SELECT_METRIC} 기준)\n'
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

    # 1) Bagging 격자
    bagging_candidates = make_bagging_candidates()
    print(f'========== 1) Bagging: {len(bagging_candidates)}개 조합 ==========', flush=True)
    run_all(bagging_candidates, X, y, cv, results)

    # 2) AdaBoost 격자
    adaboost_candidates = make_adaboost_candidates()
    print(f'========== 2) AdaBoost: {len(adaboost_candidates)}개 조합 ==========', flush=True)
    run_all(adaboost_candidates, X, y, cv, results)

    print_summary(results)

    # 3) 2.5단계 최고 설정으로 Kaggle 제출 파일 생성
    best = find_best(results, bagging_candidates + adaboost_candidates)
    make_best_submission(best, X, y, X_test, test_id)

    print(f'Total time: {(time.perf_counter() - total_start) / 60:.1f} min')
    print(f'Results: {RESULT_DIR}')


if __name__ == '__main__':
    main()
