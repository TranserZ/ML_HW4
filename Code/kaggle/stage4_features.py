"""
[Kaggle 4단계] Feature 바꾸기 (feature engineering + one-hot 압축) + ExtraTrees

- 배경: 1~3단계에서 모델 쪽 개선은 한계 (튜닝/Voting/Stacking 모두 CV 약 0.580)
        모델끼리 예측이 90% 이상 같음 -> 같은 feature로는 모두 같은 곳에서 틀림
        -> 모델이 아니라 "입력 feature"를 바꿔서 정보를 늘려 보기
- 데이터 확인 결과
    Wilderness(4열), Soil(40열) one-hot은 깨끗함 (모든 행에 1이 정확히 하나)
    노이즈는 숫자 feature 10개(고도, 거리, 경사, 방향, 그늘)에만 있음
- 비교할 feature 세트 (2 x 2 설계 -> 각 방법의 효과를 따로 볼 수 있음)
    A_original      : 원래 54개 (1~3단계와 동일, 기준)
    B_compress      : one-hot 압축 -> Wilderness 4열을 1열(번호), Soil 40열을 1열(번호)로 = 12개
                      (RF는 split마다 일부 feature만 보는데, 지금은 대부분 거의 0인 Soil 열이 뽑힘)
    C_fe            : 원래 54개 + 새 feature 12개 = 66개
    D_compress_fe   : 압축 12개 + 새 feature 12개 = 24개
- 새 feature 12개 (이 데이터 계열에서 효과가 알려진 조합, add_engineered_features 참고)
    고도 - 수직거리(수원지의 고도), 수원지까지 직선거리, 수평거리끼리의 합/차,
    Aspect(0~360도)의 sin/cos, 그늘 3개 평균
    + 노이즈 있는 feature 여러 개를 합치면 노이즈가 일부 평균되어 줄어드는 효과
- 모델
    RandomForest : 2단계 최고 설정 기반 (트리 300개, entropy). feature 개수가 바뀌므로
                   max_features, min_samples_leaf는 다시 탐색
    ExtraTrees   : RF와 같지만 split 기준값을 무작위로 정함 -> 입력 노이즈에 강함 (새 모델)
    HistGB       : 2단계 최고 설정 그대로 (확인용)
- 검증: 1~3단계와 같은 Stratified 5-fold CV (seed 42) -> 점수 그대로 비교 가능
        (A_original + RF + max_features 0.2 + min_samples_leaf 2 = 2단계의 0.5795가 다시 나와야 함)
- 결과: results/stage4/ 폴더
        results.csv            : 모든 조합의 CV 결과 (조합 하나 끝날 때마다 저장)
        best_per_config.csv    : feature 세트 x 모델별 최고 조합
        best_setting.txt       : 4단계 전체 최고 설정 (+ feature 중요도 상위 15개)
        submission_stage4_best.csv : 최고 설정을 전체 train으로 다시 학습한 제출 파일
- 중간에 멈춰도 다시 실행하면 이미 끝난 조합은 건너뛰고 이어서 진행

실행:
    cd Code/kaggle
    python -u stage4_features.py | tee results_stage4_log.txt
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / 'results' / 'stage4'

RANDOM_STATE = 42   # 1~3단계와 같은 seed -> 같은 fold
N_FOLDS = 5
CV_JOBS = 5         # 5개 fold를 동시에 학습
RF_JOBS = 4         # RF / ExtraTrees 하나가 쓰는 CPU 수 (5 fold x 4 = 20 core)

# ==================== EDIT HERE: START ====================
# 최고 설정을 고르는 기준: 'cv_acc_mean' (accuracy) 또는 'cv_f1_mean' (macro F1)
SELECT_METRIC = 'cv_acc_mean'

FEATURE_SETS = ['A_original', 'B_compress', 'C_fe', 'D_compress_fe']

# RandomForest / ExtraTrees (feature 개수가 세트마다 달라서 max_features를 넓게)
TREE_N_ESTIMATORS = 300
TREE_MAX_FEATURES = [0.2, 0.35, 0.5, 0.7]
TREE_MIN_SAMPLES_LEAF = [1, 2, 4]
# ==================== EDIT HERE: END ======================


def add_engineered_features(df):
    """숫자 feature를 조합해서 새 feature 12개 추가."""
    E = df['Elevation']
    H = df['Horizontal_Distance_To_Hydrology']
    V = df['Vertical_Distance_To_Hydrology']
    R = df['Horizontal_Distance_To_Roadways']
    F = df['Horizontal_Distance_To_Fire_Points']

    df['Elev_minus_VHyd'] = E - V                 # 가장 가까운 수원지의 고도
    df['Elev_minus_HHyd'] = E - 0.2 * H
    df['Hyd_distance'] = np.sqrt(H ** 2 + V ** 2)  # 수원지까지 직선거리
    df['Hyd_plus_Fire'] = H + F                    # 수평거리끼리의 합 / 차
    df['Hyd_minus_Fire'] = np.abs(H - F)
    df['Hyd_plus_Road'] = H + R
    df['Hyd_minus_Road'] = np.abs(H - R)
    df['Fire_plus_Road'] = F + R
    df['Fire_minus_Road'] = np.abs(F - R)
    df['Aspect_sin'] = np.sin(np.deg2rad(df['Aspect']))   # 0도와 360도는 같은 방향 -> 원 위의 좌표로
    df['Aspect_cos'] = np.cos(np.deg2rad(df['Aspect']))
    df['Hillshade_mean'] = (df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm']) / 3
    return df


def compress_one_hot(df):
    """Wilderness 4열 -> 1열(1~4번), Soil 40열 -> 1열(1~40번)."""
    wild_cols = [c for c in df.columns if c.startswith('Wilderness_Area')]
    soil_cols = [c for c in df.columns if c.startswith('Soil_Type')]
    df['Wilderness'] = df[wild_cols].to_numpy().argmax(axis=1) + 1
    df['Soil'] = df[soil_cols].to_numpy().argmax(axis=1) + 1
    return df.drop(columns=wild_cols + soil_cols)


def make_features(df, feature_set):
    df = df.copy()
    if feature_set in ['C_fe', 'D_compress_fe']:
        df = add_engineered_features(df)
    if feature_set in ['B_compress', 'D_compress_fe']:
        df = compress_one_hot(df)
    return df


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
    row.update(params)  # 파라미터별 열 (features 열로 세트별 비교)
    results.append(row)
    pd.DataFrame(results).to_csv(RESULT_DIR / 'results.csv', index=False)

    print(f'{progress}[{model_name}] {params_text}\n'
          f'    CV acc {row["cv_acc_mean"]:.4f} ± {row["cv_acc_std"]:.4f} | '
          f'CV F1 {row["cv_f1_mean"]:.4f} | train acc {row["train_acc_mean"]:.4f} | '
          f'gap {row["gap"]:.4f} | {elapsed:.1f}s', flush=True)


def find_best(results, candidates):
    """candidates 중에서 SELECT_METRIC이 가장 높은 (결과, 모델 이름, 파라미터, 모델)을 찾음."""
    best = None
    for model_name, params, model in candidates:
        for row in results:
            if row['model'] == model_name and row['params'] == params_to_text(params):
                if best is None or row[SELECT_METRIC] > best[0][SELECT_METRIC]:
                    best = (row, model_name, params, model)
    return best


def make_candidates(feature_set):
    """feature 세트 하나에 대한 (모델 이름, 파라미터, 모델) 목록."""
    candidates = []
    for max_features in TREE_MAX_FEATURES:
        for min_samples_leaf in TREE_MIN_SAMPLES_LEAF:
            params = {'features': feature_set, 'n_estimators': TREE_N_ESTIMATORS,
                      'max_features': max_features, 'min_samples_leaf': min_samples_leaf}
            model = RandomForestClassifier(
                n_estimators=TREE_N_ESTIMATORS, max_features=max_features,
                min_samples_leaf=min_samples_leaf, criterion='entropy',
                n_jobs=RF_JOBS, random_state=RANDOM_STATE)
            candidates.append(('RandomForest', params, model))

            model = ExtraTreesClassifier(
                n_estimators=TREE_N_ESTIMATORS, max_features=max_features,
                min_samples_leaf=min_samples_leaf, criterion='entropy',
                n_jobs=RF_JOBS, random_state=RANDOM_STATE)
            candidates.append(('ExtraTrees', params, model))

    # HistGB: 2단계 최고 설정 그대로
    params = {'features': feature_set}
    model = HistGradientBoostingClassifier(
        learning_rate=0.03, max_leaf_nodes=255, min_samples_leaf=5,
        l2_regularization=0.0, max_features=0.5,
        max_iter=1000, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=RANDOM_STATE)
    candidates.append(('HistGradientBoosting', params, model))
    return candidates


def print_summary(results):
    df = pd.DataFrame(results).sort_values(SELECT_METRIC, ascending=False)
    cols = ['model', 'params', 'cv_acc_mean', 'cv_acc_std', 'cv_f1_mean', 'train_acc_mean', 'gap']

    best = df.groupby(['features', 'model']).head(1).sort_values(['features', 'model'])
    best.to_csv(RESULT_DIR / 'best_per_config.csv', index=False)

    # feature 세트 x 모델 표 (보고서용)
    table = df.pivot_table(index='features', columns='model', values=SELECT_METRIC, aggfunc='max')
    print(f'\n========== feature 세트 x 모델별 최고 {SELECT_METRIC} ==========')
    print(table.to_string(float_format='%.4f'))

    if 'A_original' in table.index and 'RandomForest' in table.columns:
        reference = table.loc['A_original', 'RandomForest']
        print(f'\n========== 기준(A_original + RandomForest = {reference:.4f}) 대비 변화 ==========')
        print((table - reference).to_string(float_format='%+.4f'))

    print('\n========== 전체 상위 10개 ==========')
    print(df[cols].head(10).to_string(index=False, float_format='%.4f'))


def make_best_submission(best, X_sets, y, X_test_sets, test_id):
    """최고 설정을 전체 train으로 다시 학습해서 제출 파일 생성."""
    best_row, best_name, best_params, best_model = best
    feature_set = best_params['features']

    print(f'\n========== 4단계 최고 설정으로 제출 파일 생성 ({SELECT_METRIC} 기준) ==========')
    print(f'[{best_name}] {params_to_text(best_params)}', flush=True)

    # 5-fold CV는 train의 80%로 학습했지만, 제출용은 train 전체(100%)로 다시 학습
    best_model.fit(X_sets[feature_set], y)
    submission = pd.DataFrame({'id': test_id,
                               'Cover_Type': best_model.predict(X_test_sets[feature_set])})
    submission.to_csv(RESULT_DIR / 'submission_stage4_best.csv', index=False)

    text = (
        f'Stage 4 best ({SELECT_METRIC} 기준)\n'
        f'model: {best_name}\n'
        f'params: {params_to_text(best_params)}\n'
        f'5-fold CV accuracy: {best_row["cv_acc_mean"]:.4f} ± {best_row["cv_acc_std"]:.4f}\n'
        f'5-fold CV macro F1: {best_row["cv_f1_mean"]:.4f} ± {best_row["cv_f1_std"]:.4f}\n'
        f'train accuracy: {best_row["train_acc_mean"]:.4f} (gap {best_row["gap"]:.4f})\n'
        'Kaggle public LB: (제출 후 직접 기록)\n'
    )

    # 어떤 feature가 중요했는지 (RF / ExtraTrees만 제공)
    if hasattr(best_model, 'feature_importances_'):
        importance = pd.Series(best_model.feature_importances_, index=X_sets[feature_set].columns)
        importance = importance.sort_values(ascending=False).head(15)
        text += '\nfeature 중요도 상위 15개:\n' + importance.to_string(float_format='%.4f') + '\n'

    (RESULT_DIR / 'best_setting.txt').write_text(text, encoding='utf-8')
    print(text, flush=True)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, X_test, test_id = load_kaggle_data()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = load_results()
    total_start = time.perf_counter()

    # feature 세트별 train / test 데이터 만들기
    X_sets, X_test_sets = {}, {}
    for feature_set in FEATURE_SETS:
        X_sets[feature_set] = make_features(X, feature_set)
        X_test_sets[feature_set] = make_features(X_test, feature_set)
        print(f'{feature_set}: feature {X_sets[feature_set].shape[1]}개', flush=True)

    # feature 세트마다 RF / ExtraTrees 격자 + HistGB
    all_candidates = []
    for feature_set in FEATURE_SETS:
        candidates = make_candidates(feature_set)
        all_candidates += candidates
        print(f'========== {feature_set}: {len(candidates)}개 조합 ==========', flush=True)
        for i, (model_name, params, model) in enumerate(candidates):
            run_cv(model_name, model, params, X_sets[feature_set], y, cv, results,
                   progress=f'({i + 1}/{len(candidates)}) ')

    print_summary(results)

    # 4단계 최고 설정으로 Kaggle 제출 파일 생성
    best = find_best(results, all_candidates)
    make_best_submission(best, X_sets, y, X_test_sets, test_id)

    print(f'Total time: {(time.perf_counter() - total_start) / 60:.1f} min')
    print(f'Results: {RESULT_DIR}')


if __name__ == '__main__':
    main()
