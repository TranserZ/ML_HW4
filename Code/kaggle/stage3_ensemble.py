"""
[Kaggle 3단계] 서로 다른 알고리즘 앙상블 (Soft Voting / Stacking) + 반복 CV 최종 비교

- 목적
    (1) 1~2.5단계에서 찾은 모델별 최고 설정을 섞어서(Voting / Stacking) 더 좋아지는지 확인
    (2) fold 나누는 seed를 바꿔 3번 반복한 CV로 "단일 모델 vs 앙상블" 최종 비교표 만들기
- 방법
    1) 모델별 out-of-fold(OOF) 확률 저장
       = 5-fold에서 각 샘플이 '검증 fold'에 있을 때 받은 클래스별 확률 (학습에 안 쓴 데이터에 대한 예측)
       -> 한 번 저장하면 Voting 조합은 다시 학습하지 않고 확률 평균만으로 바로 평가 가능
    2) Soft Voting : 앙상블 모델 4개의 모든 조합(2개 이상, 11가지)을 확률 평균으로 평가
    3) Stacking    : 4개 모델의 OOF 확률(4 x 7클래스 = 28개 값)을 입력으로
                     Logistic Regression이 "어떻게 섞을지"를 학습
    4) seed 3개 x 5 fold = 15개 점수의 평균 ± std로 최종 비교
       + RF 대비 차이, 15개 fold 중 RF를 이긴 횟수 (같은 fold끼리 비교)
    5) 제출 파일: 모델마다 train 전체로 다시 학습 -> test 확률 -> 단일 모델 / 최고 Voting / Stacking
- 결과: results/stage3/ 폴더
    final_comparison.csv : 최종 비교표 (보고서용)
    agreement_seed42.csv : 모델끼리 예측이 같은 비율 (높을수록 비슷한 모델 -> 섞어도 효과 작음)
    best_setting.txt     : 3단계 최고 방법과 해당 제출 파일
    submissions/         : 제출 파일들
    cache/               : OOF 확률, test 확률 (다시 실행하면 재사용 -> 중간에 멈춰도 이어서 진행)

실행:
    cd Code/kaggle
    python -u stage3_ensemble.py | tee results_stage3_log.txt
"""
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / 'results' / 'stage3'

RANDOM_STATE = 42   # 모델 seed (1~2.5단계와 동일)
N_FOLDS = 5
CV_JOBS = 5         # 5개 fold를 동시에 학습
RF_JOBS = 4         # Random Forest / Bagging 하나가 쓰는 CPU 수

# ==================== EDIT HERE: START ====================
# 최고 방법을 고르는 기준: 'cv_acc_mean' (accuracy) 또는 'cv_f1_mean' (macro F1)
SELECT_METRIC = 'cv_acc_mean'

# fold 나누는 seed. 42 = 1~2.5단계와 같은 fold (이전 점수가 그대로 나오는지 확인용)
SEEDS = [42, 0, 1]

# Voting / Stacking에 쓸 모델 (아래 make_models의 이름)
ENSEMBLE_MEMBERS = ['RandomForest', 'Bagging', 'HistGradientBoosting', 'AdaBoost']


def make_models():
    """1~2.5단계에서 찾은 모델별 최고 설정. 주석 = seed 42 5-fold CV accuracy."""
    return {
        # 비교 기준 (단일 모델)
        'Baseline': DecisionTreeClassifier(random_state=RANDOM_STATE),                    # 0.3880
        'DecisionTree': DecisionTreeClassifier(max_depth=12, min_samples_leaf=20,
                                               random_state=RANDOM_STATE),                 # 0.5201 (1단계)
        # 앙상블 (Voting / Stacking 재료)
        'RandomForest': RandomForestClassifier(n_estimators=1000, max_features=0.2,
                                               min_samples_leaf=2, criterion='entropy',
                                               n_jobs=RF_JOBS, random_state=RANDOM_STATE),  # 0.5799 (2단계)
        'Bagging': RandomForestClassifier(n_estimators=300, max_features=1.0,
                                          min_samples_leaf=2, criterion='gini',
                                          n_jobs=RF_JOBS, random_state=RANDOM_STATE),       # 0.5762 (2.5단계)
        'HistGradientBoosting': HistGradientBoostingClassifier(
            learning_rate=0.03, max_leaf_nodes=255, min_samples_leaf=5,
            l2_regularization=0.0, max_features=0.5,
            max_iter=1000, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=RANDOM_STATE),                               # 0.5723 (2단계)
        'AdaBoost': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=20, min_samples_leaf=5,
                                             random_state=RANDOM_STATE),
            n_estimators=300, learning_rate=0.1, random_state=RANDOM_STATE),               # 0.5694 (2.5단계)
    }
# ==================== EDIT HERE: END ======================

SHORT = {'RandomForest': 'RF', 'Bagging': 'Bag', 'HistGradientBoosting': 'HGB', 'AdaBoost': 'Ada'}


def load_kaggle_data():
    train = pd.read_csv(ROOT / 'train_data.csv')
    test = pd.read_csv(ROOT / 'test_data.csv')
    X = train.drop(columns=['id', 'Cover_Type'])
    y = train['Cover_Type'].to_numpy()
    X_test = test.drop(columns='id')
    return X, y, X_test, test['id']


def make_cv(seed):
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)


def get_oof(name, model, X, y, seed):
    """5-fold OOF 확률 (N, 7). 저장된 파일이 있으면 불러옴."""
    path = RESULT_DIR / 'cache' / f'oof_{name}_seed{seed}.npy'
    if path.exists():
        return np.load(path)

    start = time.perf_counter()
    proba = cross_val_predict(model, X, y, cv=make_cv(seed), method='predict_proba', n_jobs=CV_JOBS)
    np.save(path, proba)
    print(f'  [{name}] seed {seed} OOF 완료 | {time.perf_counter() - start:.1f}s', flush=True)
    return proba


def get_test_proba(name, model, X, y, X_test):
    """train 전체로 학습한 모델의 test 확률 (20000, 7). 저장된 파일이 있으면 불러옴."""
    path = RESULT_DIR / 'cache' / f'test_proba_{name}.npy'
    if path.exists():
        return np.load(path)

    start = time.perf_counter()
    model = clone(model)
    model.fit(X, y)
    proba = model.predict_proba(X_test)
    np.save(path, proba)
    print(f'  [{name}] train 전체 학습 + test 예측 완료 | {time.perf_counter() - start:.1f}s', flush=True)
    return proba


def fold_scores(proba, y, seed):
    """확률 -> 예측 클래스 -> fold별 accuracy, macro F1."""
    classes = np.unique(y)                     # predict_proba 열 순서 = 정렬된 클래스 (1~7)
    pred = classes[np.argmax(proba, axis=1)]
    accs, f1s = [], []
    for _, valid_idx in make_cv(seed).split(np.zeros(len(y)), y):
        accs.append(accuracy_score(y[valid_idx], pred[valid_idx]))
        f1s.append(f1_score(y[valid_idx], pred[valid_idx], average='macro'))
    return accs, f1s


def stacking_oof(member_probas, y, seed):
    """Stacking의 OOF 확률: 같은 5-fold로 meta-model(Logistic Regression)을 학습/평가."""
    features = np.hstack(member_probas)        # (N, 모델 수 x 7)
    proba = np.zeros_like(member_probas[0])
    for train_idx, valid_idx in make_cv(seed).split(features, y):
        meta = LogisticRegression(max_iter=1000)
        meta.fit(features[train_idx], y[train_idx])
        proba[valid_idx] = meta.predict_proba(features[valid_idx])
    return proba


def summarize(method, n_models, probas, y, fold_accs):
    """seed별 확률 -> 15개 fold 점수 -> 평균 ± std 한 줄."""
    accs, f1s, seed_means = [], [], {}
    for seed in SEEDS:
        acc, f1 = fold_scores(probas[seed], y, seed)
        accs += acc
        f1s += f1
        seed_means[f'acc_seed{seed}'] = np.mean(acc)
    fold_accs[method] = np.array(accs)

    row = {'method': method, 'n_models': n_models,
           'cv_acc_mean': np.mean(accs), 'cv_acc_std': np.std(accs),
           'cv_f1_mean': np.mean(f1s), 'cv_f1_std': np.std(f1s)}
    row.update(seed_means)
    return row


def save_submission(proba, y, test_id, filename):
    classes = np.unique(y)
    submission = pd.DataFrame({'id': test_id, 'Cover_Type': classes[np.argmax(proba, axis=1)]})
    submission.to_csv(RESULT_DIR / 'submissions' / filename, index=False)


def main():
    (RESULT_DIR / 'cache').mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / 'submissions').mkdir(parents=True, exist_ok=True)
    X, y, X_test, test_id = load_kaggle_data()
    models = make_models()
    total_start = time.perf_counter()

    # 1) 모델별 OOF 확률 (seed마다)
    oofs = {}
    for seed in SEEDS:
        print(f'========== 1) OOF 확률: seed {seed} ==========', flush=True)
        oofs[seed] = {}
        for name, model in models.items():
            oofs[seed][name] = get_oof(name, model, X, y, seed)

    rows = []
    fold_accs = {}     # 방법별 15개 fold accuracy (RF와 fold끼리 비교용)

    # 단일 모델
    for name in models:
        probas = {}
        for seed in SEEDS:
            probas[seed] = oofs[seed][name]
        rows.append(summarize(name, 1, probas, y, fold_accs))

    # 2) Soft Voting: 2개 이상 모든 조합, 확률 평균
    voting_members = {}
    for k in range(2, len(ENSEMBLE_MEMBERS) + 1):
        for members in combinations(ENSEMBLE_MEMBERS, k):
            method = 'Voting: ' + '+'.join(SHORT[m] for m in members)
            voting_members[method] = members
            probas = {}
            for seed in SEEDS:
                probas[seed] = np.mean([oofs[seed][m] for m in members], axis=0)
            rows.append(summarize(method, k, probas, y, fold_accs))

    # 3) Stacking: 모든 앙상블 모델의 OOF 확률 -> Logistic Regression
    stacking_method = 'Stacking: ' + '+'.join(SHORT[m] for m in ENSEMBLE_MEMBERS)
    probas = {}
    for seed in SEEDS:
        probas[seed] = stacking_oof([oofs[seed][m] for m in ENSEMBLE_MEMBERS], y, seed)
    rows.append(summarize(stacking_method, len(ENSEMBLE_MEMBERS), probas, y, fold_accs))

    # 4) 최종 비교표: RF 대비 차이 + 15개 fold 중 RF를 이긴 횟수
    df = pd.DataFrame(rows).sort_values(SELECT_METRIC, ascending=False)
    df['vs_RF'] = df['cv_acc_mean'] - fold_accs['RandomForest'].mean()
    df['wins_vs_RF'] = [int(np.sum(fold_accs[m] > fold_accs['RandomForest'])) for m in df['method']]
    df.to_csv(RESULT_DIR / 'final_comparison.csv', index=False)
    print(f'\n========== 최종 비교표 (seed {SEEDS} x {N_FOLDS}-fold, {SELECT_METRIC} 순) ==========')
    print(df.to_string(index=False, float_format='%.4f'))
    print(f'* wins_vs_RF: 같은 fold끼리 비교해서 RF보다 높았던 횟수 (총 {len(SEEDS) * N_FOLDS}번)')

    # 모델끼리 예측이 같은 비율 (seed 42 OOF) -> 다양성 확인
    classes = np.unique(y)
    preds = {m: classes[np.argmax(oofs[SEEDS[0]][m], axis=1)] for m in ENSEMBLE_MEMBERS}
    agreement = pd.DataFrame(index=ENSEMBLE_MEMBERS, columns=ENSEMBLE_MEMBERS, dtype=float)
    for a in ENSEMBLE_MEMBERS:
        for b in ENSEMBLE_MEMBERS:
            agreement.loc[a, b] = np.mean(preds[a] == preds[b])
    agreement.to_csv(RESULT_DIR / 'agreement_seed42.csv')
    print('\n========== 모델끼리 예측이 같은 비율 (seed 42) ==========')
    print(agreement.round(3).to_string())

    # 5) 제출 파일: train 전체로 다시 학습한 모델의 test 확률 사용
    print('\n========== 5) 제출 파일 생성 ==========', flush=True)
    test_probas = {}
    submission_files = {}
    for name, model in models.items():
        test_probas[name] = get_test_proba(name, model, X, y, X_test)
        save_submission(test_probas[name], y, test_id, f'submission_{name}.csv')
        submission_files[name] = f'submission_{name}.csv'

    # 최고 Voting 조합
    best_voting = df[df['method'].str.startswith('Voting')].iloc[0]['method']
    members = voting_members[best_voting]
    save_submission(np.mean([test_probas[m] for m in members], axis=0), y, test_id,
                    'submission_voting_best.csv')
    submission_files[best_voting] = 'submission_voting_best.csv'

    # Stacking: seed 42 OOF 확률 전체로 meta-model 학습 -> test 확률에 적용
    meta = LogisticRegression(max_iter=1000)
    meta.fit(np.hstack([oofs[SEEDS[0]][m] for m in ENSEMBLE_MEMBERS]), y)
    stacking_test = meta.predict_proba(np.hstack([test_probas[m] for m in ENSEMBLE_MEMBERS]))
    save_submission(stacking_test, y, test_id, 'submission_stacking.csv')
    submission_files[stacking_method] = 'submission_stacking.csv'

    # 3단계 최고 방법 기록
    best = df.iloc[0]
    text = (
        f'Stage 3 best ({SELECT_METRIC} 기준, seed {SEEDS} x {N_FOLDS}-fold)\n'
        f'method: {best["method"]}\n'
        f'CV accuracy: {best["cv_acc_mean"]:.4f} ± {best["cv_acc_std"]:.4f}\n'
        f'CV macro F1: {best["cv_f1_mean"]:.4f} ± {best["cv_f1_std"]:.4f}\n'
        f'RF 대비: {best["vs_RF"]:+.4f} ({len(SEEDS) * N_FOLDS}개 fold 중 {best["wins_vs_RF"]}번 RF보다 높음)\n'
        f'submission: submissions/{submission_files[best["method"]]}\n'
        'Kaggle public LB: (제출 후 직접 기록)\n'
    )
    (RESULT_DIR / 'best_setting.txt').write_text(text, encoding='utf-8')
    print('\n' + text)
    print(f'Total time: {(time.perf_counter() - total_start) / 60:.1f} min')
    print(f'Results: {RESULT_DIR}')


if __name__ == '__main__':
    main()
