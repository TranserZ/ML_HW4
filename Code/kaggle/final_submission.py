"""
[Kaggle 최종 제출 코드] Feature engineering + ExtraTrees (+ 노이즈 레이블 제거)

- 이 파일 하나로 최종 Kaggle 제출 파일을 처음부터 똑같이 다시 만듦
  (stage4_5_noise_filter.py가 최종 선택한 설정과 같은 결과)
- 필요한 파일: 같은 폴더의 train_data.csv, test_data.csv
- 순서
    1) Feature engineering: 숫자 feature를 조합해서 새 feature 12개 추가 (54개 -> 66개)
    2) 노이즈 레이블 제거 (REMOVE_FRACTION > 0일 때만)
       - train을 5-fold로 나눠서, 각 샘플이 학습에 안 쓰였을 때 모델이 '자기 레이블'에 준 확률을 구함
       - 이 확률이 가장 낮은 샘플부터 REMOVE_FRACTION 비율만큼 제거 (노이즈 레이블일 가능성이 높음)
    3) 남은 train 전체로 ExtraTrees 학습 -> test 예측 -> submission_final.csv
- 재현성
    - 모든 random_state를 42로 고정
    - 여러 CPU로 예측하면 트리별 확률을 더하는 순서가 매번 달라서 1e-16 정도의 오차가 생기고,
      확률이 똑같은(동점) 클래스끼리 예측이 바뀔 수 있음 -> 확률을 반올림한 뒤 예측

실행:
    cd Code/kaggle
    python final_submission.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


ROOT = Path(__file__).resolve().parent
OUTPUT_CSV = ROOT / 'submission_final.csv'
# 4.5단계에서 만든 제출 파일 (있으면 결과가 완전히 같은지 자동 확인)
STAGE4_5_CSV = ROOT / 'results' / 'stage4_5' / 'submission_stage4_5_best.csv'

RANDOM_STATE = 42
N_FOLDS = 5
N_JOBS = -1         # CPU 수는 결과에 영향 없음 (random_state 고정 + 확률 반올림)

# ==================== EDIT HERE: START ====================
# 4.5단계 최종 선택 (results/stage4_5/best_setting.txt)
#   ET + 노이즈 25% 제거, 반복 CV accuracy 0.6031 ± 0.0031 (seed 3개 x 5 fold)
MODEL_TYPE = 'ExtraTrees'
N_ESTIMATORS = 300
MAX_FEATURES = 0.7
MIN_SAMPLES_LEAF = 2
REMOVE_FRACTION = 0.25
# 노이즈 판별용 모델 = 4.5단계 (a)의 최고 설정
FILTER_N_ESTIMATORS = 300
FILTER_MAX_FEATURES = 0.7
FILTER_MIN_SAMPLES_LEAF = 3
# ==================== EDIT HERE: END ======================


def load_kaggle_data():
    train = pd.read_csv(ROOT / 'train_data.csv')
    test = pd.read_csv(ROOT / 'test_data.csv')
    X = train.drop(columns=['id', 'Cover_Type'])
    y = train['Cover_Type']
    X_test = test.drop(columns='id')
    return X, y, X_test, test['id']


def add_engineered_features(df):
    """숫자 feature를 조합해서 새 feature 12개 추가 (4단계 C_fe와 같음)."""
    df = df.copy()
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


def make_model(model_type, n_estimators, max_features, min_samples_leaf):
    model_class = ExtraTreesClassifier if model_type == 'ExtraTrees' else RandomForestClassifier
    return model_class(n_estimators=n_estimators, max_features=max_features,
                       min_samples_leaf=min_samples_leaf, criterion='entropy',
                       n_jobs=N_JOBS, random_state=RANDOM_STATE)


def stable_proba(proba):
    """확률을 반올림해서 실행할 때마다 똑같은 예측이 나오게 함."""
    return np.round(proba, 10)


def remove_noisy_labels(X, y):
    """5-fold OOF 확률로 '자기 레이블' 확률이 가장 낮은 샘플을 REMOVE_FRACTION 비율만큼 제거."""
    filter_model = make_model('ExtraTrees', FILTER_N_ESTIMATORS, FILTER_MAX_FEATURES,
                              FILTER_MIN_SAMPLES_LEAF)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    proba = stable_proba(cross_val_predict(filter_model, X, y, cv=cv, method='predict_proba'))

    classes = np.sort(y.unique())                       # proba 열 순서 = 정렬된 클래스 순서
    own = proba[np.arange(len(y)), np.searchsorted(classes, y)]

    n_remove = int(round(len(y) * REMOVE_FRACTION))
    order = np.argsort(own, kind='stable')              # 확률이 낮은 순서
    keep = np.sort(order[n_remove:])

    removed = y.drop(index=y.index[keep])
    print(f'노이즈 의심 샘플 제거: {len(removed)}개 / {len(y)}개')
    print('클래스별 제거 비율: ' + ', '.join(
        f'{c}: {(removed == c).sum() / (y == c).sum():.1%}' for c in classes))
    return X.iloc[keep], y.iloc[keep]


def main():
    X, y, X_test, test_id = load_kaggle_data()
    X = add_engineered_features(X)
    X_test = add_engineered_features(X_test)
    print(f'feature {X.shape[1]}개, train {len(y)}개, test {len(X_test)}개')

    if REMOVE_FRACTION > 0:
        X, y = remove_noisy_labels(X, y)

    print(f'{MODEL_TYPE} 학습 (n_estimators={N_ESTIMATORS}, max_features={MAX_FEATURES}, '
          f'min_samples_leaf={MIN_SAMPLES_LEAF}, 학습 샘플 {len(y)}개)')
    model = make_model(MODEL_TYPE, N_ESTIMATORS, MAX_FEATURES, MIN_SAMPLES_LEAF).fit(X, y)
    pred = model.classes_[stable_proba(model.predict_proba(X_test)).argmax(axis=1)]

    submission = pd.DataFrame({'id': test_id, 'Cover_Type': pred})
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f'저장: {OUTPUT_CSV}')
    print('test 예측 클래스 비율: ' + ', '.join(
        f'{c}: {(pred == c).mean():.1%}' for c in model.classes_))

    # 4.5단계 제출 파일과 완전히 같은지 확인
    if STAGE4_5_CSV.exists():
        reference = pd.read_csv(STAGE4_5_CSV)
        n_diff = int((reference['Cover_Type'].to_numpy() != pred).sum())
        if n_diff == 0 and reference['id'].equals(test_id):
            print('확인: 4.5단계 제출 파일과 완전히 같음')
        else:
            print(f'주의: 4.5단계 제출 파일과 {n_diff}개 행이 다름 -> EDIT HERE 설정을 '
                  'results/stage4_5/best_setting.txt와 비교해 보기')


if __name__ == '__main__':
    main()
