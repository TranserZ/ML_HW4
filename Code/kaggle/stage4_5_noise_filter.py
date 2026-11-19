"""
[Kaggle 4.5단계] ExtraTrees 범위 확장 + 노이즈 레이블 제거 + 반복 CV 최종 비교

- 배경
    4단계 최고 = ExtraTrees + C_fe (max_features 0.7, min_samples_leaf 4) -> CV 0.6002
    그런데 이 설정이 탐색 범위의 끝 (max_features 최댓값, leaf 최댓값) -> 범위를 넓혀 확인 필요
    train 레이블의 약 18%가 노이즈로 추정됨 (CV 약 0.58 vs LB 약 0.70에서 계산)
- 순서 (feature는 C_fe, 모델은 ExtraTrees로 고정)
    (a) 범위 확장: max_features [0.7, 0.85, 1.0] x min_samples_leaf [3, 4, 6, 8, 12]
        + 최고 설정에서 트리 수 600 / 1000 확인
        (max_features 0.7 + leaf 4는 4단계와 같은 조합 -> 0.6002가 다시 나와야 함)
    (b) 노이즈 레이블 제거 ((a)의 최고 설정 위에서)
        1) 각 fold의 학습 데이터(80%) 안에서 다시 5-fold로 OOF 확률 계산
           = 각 샘플이 학습에 안 쓰였을 때 모델이 "자기 레이블"에 준 확률
        2) 이 확률이 가장 낮은 샘플부터 일정 비율(0~30%) 제거 (노이즈 레이블일 가능성이 높은 샘플)
        3) 남은 데이터로 학습하고, 검증 fold(20%)는 그대로 평가 (검증 데이터는 절대 건드리지 않음)
        + 제거 후에는 데이터가 깨끗해져서 더 작은 leaf가 나을 수도 있음 -> leaf도 같이 확인
        (제거 비율 0% = (a)의 최고 설정과 완전히 같은 결과가 나와야 함)
    (c) 반복 CV 최종 비교: fold 나누는 seed 3개 x 5 fold = 15개 점수 (3단계와 같은 방식)
        4단계 RF / 4단계 ET / (a) 최고 / (b) 최고 + 3단계 결과(Baseline, RF, Stacking 등)를 한 표로
        -> 15개 점수 평균이 가장 높은 4~4.5단계 설정으로 제출 파일 생성
- 결과: results/stage4_5/ 폴더
    results.csv             : (a), (b) 모든 조합의 CV 결과 (seed 42, 조합 하나 끝날 때마다 저장)
    removed_by_class.csv    : 제거 비율별로 각 클래스의 레이블이 몇 % 제거됐는지
    repeated_cv.csv         : (c)의 fold별 점수
    final_comparison.csv    : (c) 최종 비교표 (보고서용)
    best_setting.txt        : 최종 선택 설정
    submission_stage4_5_best.csv : 최종 선택 설정을 전체 train으로 다시 학습한 제출 파일
    cache/                  : OOF 확률 (다시 실행하면 재사용 -> 중간에 멈춰도 이어서 진행)

실행:
    cd Code/kaggle
    python -u stage4_5_noise_filter.py | tee results_stage4_5_log.txt
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# feature는 4단계와 완전히 같은 함수를 그대로 사용
from stage4_features import make_features, load_kaggle_data, params_to_text


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / 'results' / 'stage4_5'
CACHE_DIR = RESULT_DIR / 'cache'

RANDOM_STATE = 42   # 모델 seed (1~4단계와 동일)
N_FOLDS = 5
CV_JOBS = 5         # 5개 fold를 동시에 학습
RF_JOBS = 4         # 트리 모델 하나가 쓰는 CPU 수 (5 fold x 4 = 20 core)
FIT_JOBS = CV_JOBS * RF_JOBS   # 모델 하나만 학습할 때 쓰는 CPU 수
FEATURE_SET = 'C_fe'           # 4단계 최고 feature 세트

# ==================== EDIT HERE: START ====================
# 최고 설정을 고르는 기준: 'cv_acc_mean' (accuracy) 또는 'cv_f1_mean' (macro F1)
SELECT_METRIC = 'cv_acc_mean'

# (a) 범위 확장
ET_N_ESTIMATORS = 300
ET_MAX_FEATURES = [0.7, 0.85, 1.0]
ET_MIN_SAMPLES_LEAF = [3, 4, 6, 8, 12]
MORE_TREES = [600, 1000]        # (a) 최고 설정에서 트리 수만 늘려 확인

# (b) 노이즈 레이블 제거
REMOVE_FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
FILTER_EXTRA_LEAF = [2]         # 제거 후 추가로 확인할 leaf ((a)의 최고 leaf는 자동 포함)

# (c) 반복 CV (42 = 1~4단계와 같은 fold)
SEEDS = [42, 0, 1]

# 4단계 최고 설정 (비교 기준)
STAGE4_RF = {'n_estimators': 300, 'max_features': 0.2, 'min_samples_leaf': 2}
STAGE4_ET = {'n_estimators': 300, 'max_features': 0.7, 'min_samples_leaf': 4}
# ==================== EDIT HERE: END ======================


def make_model(model_type, n_estimators, max_features, min_samples_leaf, n_jobs=RF_JOBS):
    model_class = ExtraTreesClassifier if model_type == 'ExtraTrees' else RandomForestClassifier
    return model_class(n_estimators=n_estimators, max_features=max_features,
                       min_samples_leaf=min_samples_leaf, criterion='entropy',
                       n_jobs=n_jobs, random_state=RANDOM_STATE)


def load_results(name):
    path = RESULT_DIR / name
    if path.exists():
        return pd.read_csv(path).to_dict('records')
    return []


def is_done(results, model_name, params):
    params_text = params_to_text(params)
    return any(row['model'] == model_name and row['params'] == params_text for row in results)


def save_row(model_name, params, accs, f1s, train_accs, elapsed, results, progress=''):
    """fold별 점수를 평균 내서 results에 추가 + 저장."""
    row = {
        'model': model_name,
        'params': params_to_text(params),
        'cv_acc_mean': np.mean(accs),
        'cv_acc_std': np.std(accs),
        'cv_f1_mean': np.mean(f1s),
        'cv_f1_std': np.std(f1s),
        'train_acc_mean': np.mean(train_accs),
        'gap': np.mean(train_accs) - np.mean(accs),   # 클수록 노이즈를 외움
        'time_sec': elapsed,
    }
    row.update(params)
    results.append(row)
    pd.DataFrame(results).to_csv(RESULT_DIR / 'results.csv', index=False)

    print(f'{progress}[{model_name}] {row["params"]}\n'
          f'    CV acc {row["cv_acc_mean"]:.4f} ± {row["cv_acc_std"]:.4f} | '
          f'CV F1 {row["cv_f1_mean"]:.4f} | train acc {row["train_acc_mean"]:.4f} | '
          f'gap {row["gap"]:.4f} | {elapsed:.1f}s', flush=True)


def stable_proba(proba):
    """여러 CPU로 예측하면 트리별 확률을 더하는 순서가 매번 달라서 아주 작은 오차(1e-16)가 생김
    -> 확률이 똑같은 클래스(동점)끼리 예측이 실행할 때마다 바뀔 수 있음
    -> 반올림해서 항상 같은 결과가 나오게 함 (제출 파일 재현용)"""
    return np.round(proba, 10)


def stable_predict(model, X):
    return model.classes_[stable_proba(model.predict_proba(X)).argmax(axis=1)]


# ---------------------------- 노이즈 레이블 제거 ----------------------------
def filter_tag(filter_params):
    return (f'n{filter_params["n_estimators"]}_mf{filter_params["max_features"]}'
            f'_leaf{filter_params["min_samples_leaf"]}')


def own_label_proba(filter_params, X, y, seed, cache_name):
    """X, y 안에서 5-fold OOF 확률을 구해서, 각 샘플이 '자기 레이블'에 받은 확률을 반환."""
    path = CACHE_DIR / f'own_proba_{filter_tag(filter_params)}_{cache_name}.npy'
    if path.exists():
        return np.load(path)

    inner_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    proba = cross_val_predict(make_model('ExtraTrees', **filter_params), X, y, cv=inner_cv,
                              method='predict_proba', n_jobs=CV_JOBS)
    proba = stable_proba(proba)
    classes = np.sort(y.unique())                       # proba 열 순서 = 정렬된 클래스 순서
    own = proba[np.arange(len(y)), np.searchsorted(classes, y)]
    np.save(path, own)
    return own


def keep_index(own, remove_fraction):
    """자기 레이블 확률이 낮은 샘플부터 remove_fraction 비율만큼 빼고 남길 위치."""
    n_remove = int(round(len(own) * remove_fraction))
    order = np.argsort(own, kind='stable')              # 확률이 낮은 순서
    return np.sort(order[n_remove:])


def cv_scores(model_type, params, X, y, seed, filter_params=None, remove_fraction=0.0):
    """5-fold CV -> fold별 (accuracy, macro F1, train accuracy).
    remove_fraction > 0이면 각 fold의 학습 데이터에서만 노이즈 의심 샘플을 빼고 학습 (검증 fold는 그대로)."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    accs, f1s, train_accs = [], [], []
    for k, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        if remove_fraction > 0:
            own = own_label_proba(filter_params, X_train, y_train, seed, f'seed{seed}_fold{k}')
            keep = keep_index(own, remove_fraction)
            X_train, y_train = X_train.iloc[keep], y_train.iloc[keep]

        model = make_model(model_type, **params, n_jobs=FIT_JOBS).fit(X_train, y_train)
        pred = stable_predict(model, X.iloc[valid_idx])
        accs.append(accuracy_score(y.iloc[valid_idx], pred))
        f1s.append(f1_score(y.iloc[valid_idx], pred, average='macro'))
        train_accs.append(accuracy_score(y_train, stable_predict(model, X_train)))  # 학습에 쓴 데이터 기준
    return np.array(accs), np.array(f1s), np.array(train_accs)


def save_removed_by_class(filter_params, X, y):
    """제거 비율별로 각 클래스 레이블의 몇 %가 제거됐는지 (seed 42, 5개 fold 평균)."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    classes = np.sort(y.unique())
    rows = []
    for remove_fraction in REMOVE_FRACTIONS:
        if remove_fraction == 0:
            continue
        ratio = np.zeros(len(classes))
        for k, (train_idx, _) in enumerate(cv.split(X, y)):
            y_train = y.iloc[train_idx].to_numpy()
            own = own_label_proba(filter_params, X.iloc[train_idx], y.iloc[train_idx],
                                  RANDOM_STATE, f'seed{RANDOM_STATE}_fold{k}')
            removed = np.ones(len(y_train), dtype=bool)
            removed[keep_index(own, remove_fraction)] = False
            ratio += [removed[y_train == c].mean() for c in classes]
        rows.append({'remove_fraction': remove_fraction,
                     **{f'class_{c}': r for c, r in zip(classes, ratio / N_FOLDS)}})

    table = pd.DataFrame(rows)
    table.to_csv(RESULT_DIR / 'removed_by_class.csv', index=False)
    print('\n========== 제거 비율별 각 클래스 레이블의 제거 비율 (seed 42, fold 평균) ==========')
    print('(train 레이블 비율: ' + ', '.join(f'{c}: {(y == c).mean():.1%}' for c in classes) + ')')
    print(table.to_string(index=False, float_format=lambda v: f'{v:.1%}'))


# ---------------------------- 단계별 실행 ----------------------------
def best_of(results, model_name):
    rows = [row for row in results if row['model'] == model_name]
    return max(rows, key=lambda row: row[SELECT_METRIC])


def tree_params(row):
    return {'n_estimators': int(row['n_estimators']), 'max_features': float(row['max_features']),
            'min_samples_leaf': int(row['min_samples_leaf'])}


def part_a_range(X, y, results):
    print('\n==================== (a) ExtraTrees 범위 확장 ====================', flush=True)
    grid = [{'n_estimators': ET_N_ESTIMATORS, 'max_features': mf, 'min_samples_leaf': leaf}
            for mf in ET_MAX_FEATURES for leaf in ET_MIN_SAMPLES_LEAF]

    for i, params in enumerate(grid):
        params = {'features': FEATURE_SET, **params}
        if not is_done(results, 'ExtraTrees', params):
            start = time.perf_counter()
            scores = cv_scores('ExtraTrees', tree_params(params), X[FEATURE_SET], y, RANDOM_STATE)
            save_row('ExtraTrees', params, *scores, time.perf_counter() - start, results,
                     progress=f'({i + 1}/{len(grid)}) ')

    # 최고 설정에서 트리 수만 늘려 보기
    best = tree_params(best_of(results, 'ExtraTrees'))
    for n_estimators in MORE_TREES:
        params = {'features': FEATURE_SET, **best, 'n_estimators': n_estimators}
        if not is_done(results, 'ExtraTrees', params):
            start = time.perf_counter()
            scores = cv_scores('ExtraTrees', tree_params(params), X[FEATURE_SET], y, RANDOM_STATE)
            save_row('ExtraTrees', params, *scores, time.perf_counter() - start, results,
                     progress='(트리 수 확인) ')

    df = pd.DataFrame([row for row in results if row['model'] == 'ExtraTrees'
                       and row['n_estimators'] == ET_N_ESTIMATORS])
    print(f'\n========== (a) max_features x min_samples_leaf ({ET_N_ESTIMATORS} trees, {SELECT_METRIC}) ==========')
    print(df.pivot_table(index='max_features', columns='min_samples_leaf',
                         values=SELECT_METRIC).to_string(float_format='%.4f'))
    best_row = best_of(results, 'ExtraTrees')
    print(f'(a) 최고: {best_row["params"]} -> {best_row[SELECT_METRIC]:.4f}', flush=True)
    return tree_params(best_row)


def part_b_filter(X, y, results, best_a):
    print('\n==================== (b) 노이즈 레이블 제거 ====================', flush=True)
    # 노이즈 판별용 모델 = (a) 최고 설정 (트리 수는 속도 때문에 ET_N_ESTIMATORS)
    filter_params = {**best_a, 'n_estimators': ET_N_ESTIMATORS}
    leaf_options = sorted(set([best_a['min_samples_leaf']] + FILTER_EXTRA_LEAF))
    print(f'노이즈 판별 모델: ExtraTrees {params_to_text(filter_params)}', flush=True)

    combos = [(leaf, fraction) for leaf in leaf_options for fraction in REMOVE_FRACTIONS]
    for i, (leaf, remove_fraction) in enumerate(combos):
        params = {'features': FEATURE_SET, **best_a, 'min_samples_leaf': leaf,
                  'remove_fraction': remove_fraction, 'filter_model': filter_tag(filter_params)}
        if not is_done(results, 'ExtraTrees+filter', params):
            start = time.perf_counter()
            scores = cv_scores('ExtraTrees', tree_params(params), X[FEATURE_SET], y, RANDOM_STATE,
                               filter_params, remove_fraction)
            save_row('ExtraTrees+filter', params, *scores, time.perf_counter() - start, results,
                     progress=f'({i + 1}/{len(combos)}) ')

    df = pd.DataFrame([row for row in results if row['model'] == 'ExtraTrees+filter'
                       and row['filter_model'] == filter_tag(filter_params)])
    table = df.pivot_table(index='remove_fraction', columns='min_samples_leaf', values=SELECT_METRIC)
    print(f'\n========== (b) 제거 비율 x min_samples_leaf ({SELECT_METRIC}) ==========')
    print(table.to_string(float_format='%.4f'))
    print('\n========== (b) 제거 안 함(0%) 대비 변화 ==========')
    print((table - table.loc[0.0]).to_string(float_format='%+.4f'))

    save_removed_by_class(filter_params, X[FEATURE_SET], y)

    best_row = max(df.to_dict('records'), key=lambda row: row[SELECT_METRIC])
    print(f'(b) 최고: {best_row["params"]} -> {best_row[SELECT_METRIC]:.4f}', flush=True)
    return filter_params, tree_params(best_row), float(best_row['remove_fraction'])


def part_c_repeated(X, y, candidates):
    print('\n==================== (c) 반복 CV 최종 비교 ====================', flush=True)
    fold_rows = load_results('repeated_cv.csv')
    for name, model_type, params, filter_params, remove_fraction in candidates:
        for seed in SEEDS:
            if any(row['name'] == name and row['seed'] == seed for row in fold_rows):
                continue
            start = time.perf_counter()
            accs, f1s, _ = cv_scores(model_type, params, X, y, seed, filter_params, remove_fraction)
            for k, (acc, f1) in enumerate(zip(accs, f1s)):
                fold_rows.append({'name': name, 'seed': seed, 'fold': k, 'acc': acc, 'f1': f1})
            pd.DataFrame(fold_rows).to_csv(RESULT_DIR / 'repeated_cv.csv', index=False)
            print(f'[{name}] seed {seed}: acc {np.mean(accs):.4f} | F1 {np.mean(f1s):.4f} '
                  f'| {time.perf_counter() - start:.1f}s', flush=True)

    folds = pd.DataFrame(fold_rows)
    names = [c[0] for c in candidates]
    reference = folds[folds.name == names[1]].sort_values(['seed', 'fold'])   # 4단계 ET

    rows = []
    for name, model_type, params, _, remove_fraction in candidates:
        f = folds[folds.name == name].sort_values(['seed', 'fold'])
        setting = params_to_text(params) + (f', remove_fraction={remove_fraction}' if remove_fraction else '')
        rows.append({
            'method': name, 'setting': setting,
            'cv_acc_mean': f.acc.mean(), 'cv_acc_std': f.acc.std(ddof=0),
            'cv_f1_mean': f.f1.mean(), 'cv_f1_std': f.f1.std(ddof=0),
            **{f'acc_seed{s}': f[f.seed == s].acc.mean() for s in SEEDS},
            'vs_stage4_ET': f.acc.mean() - reference.acc.mean(),
            'wins_vs_stage4_ET': int((f.acc.to_numpy() > reference.acc.to_numpy()).sum()),
        })

    # 3단계 결과(원래 feature, 같은 seed / fold)도 같은 표에 붙이기
    stage3_csv = ROOT / 'results' / 'stage3' / 'final_comparison.csv'
    if stage3_csv.exists():
        stage3 = pd.read_csv(stage3_csv)
        stage3_rows = pd.concat([
            stage3[stage3.method.isin(['Baseline', 'DecisionTree', 'RandomForest'])],
            stage3[stage3.method.str.startswith('Stacking')].head(1),
            stage3[stage3.method.str.startswith('Voting')].sort_values(SELECT_METRIC, ascending=False).head(1),
        ])
        for _, row in stage3_rows.iterrows():
            rows.append({'method': f'{row["method"]} (A_original, 3단계)', 'setting': '',
                         **{col: row[col] for col in ['cv_acc_mean', 'cv_acc_std', 'cv_f1_mean', 'cv_f1_std']
                            + [f'acc_seed{s}' for s in SEEDS]}})

    table = pd.DataFrame(rows).sort_values(SELECT_METRIC, ascending=False)
    table.to_csv(RESULT_DIR / 'final_comparison.csv', index=False)
    print(f'\n========== 최종 비교 (seed {len(SEEDS)}개 x {N_FOLDS} fold = {len(SEEDS) * N_FOLDS}개 점수) ==========')
    print(table.drop(columns='setting').to_string(index=False, float_format='%.4f'))
    return table


def make_submission(choice, X, y, X_test, test_id, table):
    """최종 선택 설정을 전체 train으로 다시 학습해서 제출 파일 생성."""
    name, model_type, params, filter_params, remove_fraction = choice
    print(f'\n========== 최종 선택: {name} -> 제출 파일 생성 ==========', flush=True)

    X_train, y_train = X, y
    removed_text = ''
    if remove_fraction > 0:
        # 전체 train 안에서 5-fold OOF 확률 -> 노이즈 의심 샘플 제거 (CV 때와 같은 방법)
        own = own_label_proba(filter_params, X, y, RANDOM_STATE, 'full_train')
        keep = keep_index(own, remove_fraction)
        X_train, y_train = X.iloc[keep], y.iloc[keep]
        removed = y.drop(index=y.index[keep])
        removed_text = (f'제거한 샘플: {len(removed)}개 / {len(y)}개\n클래스별 제거 비율: '
                        + ', '.join(f'{c}: {(removed == c).sum() / (y == c).sum():.1%}'
                                    for c in np.sort(y.unique())) + '\n')

    model = make_model(model_type, **params, n_jobs=FIT_JOBS).fit(X_train, y_train)
    pred = stable_predict(model, X_test)
    submission =pd.DataFrame({'id': test_id, 'Cover_Type': pred})
    submission.to_csv(RESULT_DIR / 'submission_stage4_5_best.csv', index=False)

    row = table[table.method == name].iloc[0]
    text = (
        f'Stage 4.5 final choice ({SELECT_METRIC}, seed {len(SEEDS)}개 x {N_FOLDS} fold 평균 기준)\n'
        f'method: {name}\n'
        f'model: {model_type}, features={FEATURE_SET}\n'
        f'setting: {row["setting"]}\n'
        f'반복 CV accuracy: {row["cv_acc_mean"]:.4f} ± {row["cv_acc_std"]:.4f}\n'
        f'반복 CV macro F1: {row["cv_f1_mean"]:.4f} ± {row["cv_f1_std"]:.4f}\n'
        f'4단계 ET 대비: {row["vs_stage4_ET"]:+.4f} ({int(row["wins_vs_stage4_ET"])}/{len(SEEDS) * N_FOLDS} fold 승)\n'
        + removed_text +
        'Kaggle public LB: (제출 후 직접 기록)\n'
    )

    stage4_csv = ROOT / 'results' / 'stage4' / 'submission_stage4_best.csv'
    if stage4_csv.exists():
        stage4_pred = pd.read_csv(stage4_csv)['Cover_Type'].to_numpy()
        text += f'4단계 제출 파일과 test 예측이 다른 비율: {(stage4_pred != pred).mean():.1%}\n'
    text += ('test 예측 클래스 비율: '
             + ', '.join(f'{c}: {(pred == c).mean():.1%}' for c in np.sort(y.unique())) + '\n')

    (RESULT_DIR / 'best_setting.txt').write_text(text, encoding='utf-8')
    print(text, flush=True)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    X_raw, y, X_test_raw, test_id = load_kaggle_data()
    X = {FEATURE_SET: make_features(X_raw, FEATURE_SET)}
    X_test = make_features(X_test_raw, FEATURE_SET)
    print(f'{FEATURE_SET}: feature {X[FEATURE_SET].shape[1]}개', flush=True)

    results = load_results('results.csv')
    total_start = time.perf_counter()

    best_a = part_a_range(X, y, results)
    filter_params, best_b, best_fraction = part_b_filter(X, y, results, best_a)

    # (c) 후보: 4단계 RF, 4단계 ET, (a) 최고, (b) 최고 (같은 설정이면 하나만)
    candidates = [
        ('RF (C_fe, 4단계)', 'RandomForest', STAGE4_RF, None, 0.0),
        ('ET (C_fe, 4단계)', 'ExtraTrees', STAGE4_ET, None, 0.0),
    ]
    if best_a != STAGE4_ET:
        candidates.append(('ET 범위 확장 (4.5단계 a)', 'ExtraTrees', best_a, None, 0.0))
    if best_fraction > 0:
        candidates.append((f'ET + 노이즈 {best_fraction:.0%} 제거 (4.5단계 b)', 'ExtraTrees',
                           best_b, filter_params, best_fraction))
    elif best_b != best_a:
        candidates.append(('ET leaf 변경 (4.5단계 b)', 'ExtraTrees', best_b, None, 0.0))

    table = part_c_repeated(X[FEATURE_SET], y, candidates)

    # 15개 점수 평균이 가장 높은 4~4.5단계 설정으로 제출
    names = [c[0] for c in candidates]
    best_name = table[table.method.isin(names)].iloc[0]['method']
    choice = candidates[names.index(best_name)]
    make_submission(choice, X[FEATURE_SET], y, X_test, test_id, table)

    print(f'Total time: {(time.perf_counter() - total_start) / 60:.1f} min')
    print(f'Results: {RESULT_DIR}')


if __name__ == '__main__':
    main()
