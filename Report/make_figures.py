"""
Kaggle 리포트용 그래프 생성 (Code/kaggle/results/의 결과 CSV를 읽어서 Report/figures/에 PNG 저장)

실행:
    cd Report
    python make_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT.parent / 'Code' / 'kaggle'
RESULTS = KAGGLE / 'results'
FIG_DIR = ROOT / 'figures'

# ---------------------------- 스타일 ----------------------------
SURFACE = '#ffffff'
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
GRID = '#e4e3df'
MUTED = '#b4b2ab'          # 강조하지 않는 막대 / 점
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'   # 범주 색 1~3번 (항상 이 순서)
SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
MODEL_COLOR = {'ExtraTrees': BLUE, 'RandomForest': ORANGE, 'HistGradientBoosting': AQUA}

KOREAN_FONTS = [
    '/mnt/c/Windows/Fonts/malgun.ttf',                         # WSL (Windows 맑은 고딕)
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',          # Ubuntu: sudo apt install fonts-nanum
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    'C:/Windows/Fonts/malgun.ttf',
    '/System/Library/Fonts/AppleSDGothicNeo.ttc',
]


def setup_style():
    for path in KOREAN_FONTS:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=path).get_name()
            break
    else:
        print('주의: 한글 폰트를 찾지 못함 -> 한글이 네모로 보일 수 있음 (sudo apt install fonts-nanum)')

    plt.rcParams.update({
        'axes.unicode_minus': False,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'text.color': TEXT, 'axes.labelcolor': TEXT_2, 'xtick.color': TEXT_2, 'ytick.color': TEXT_2,
        'axes.edgecolor': GRID, 'axes.linewidth': 1.0,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.linestyle': '-',
        'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 9, 'legend.frameon': False,
        'lines.linewidth': 2, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    })


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200)
    plt.close(fig)
    print('저장:', FIG_DIR / name)


def line(ax, x, y, color, label=None, **kwargs):
    """2px 선 + 테두리(흰색 링) 있는 점."""
    ax.plot(x, y, color=color, label=label, marker='o', markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, **kwargs)


def label_bars(ax, bars, fmt, horizontal=False, pad=3):
    for bar in bars:
        value = bar.get_width() if horizontal else bar.get_height()
        if horizontal:
            ax.annotate(fmt(value), (value, bar.get_y() + bar.get_height() / 2), xytext=(pad, 0),
                        textcoords='offset points', va='center', ha='left', color=TEXT, fontsize=9)
        else:
            ax.annotate(fmt(value), (bar.get_x() + bar.get_width() / 2, value), xytext=(0, pad),
                        textcoords='offset points', ha='center', va='bottom', color=TEXT, fontsize=9)


def heatmap(ax, table, fmt, cmap_colors=SEQ_BLUE):
    cmap = LinearSegmentedColormap.from_list('seq', cmap_colors)
    values = table.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, aspect='auto')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    low, high = np.nanmin(values), np.nanmax(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            dark_cell = (values[i, j] - low) / (high - low + 1e-12) > 0.55
            ax.text(j, i, fmt(values[i, j]), ha='center', va='center', fontsize=9,
                    color='white' if dark_cell else TEXT)
    ax.set_xticks(range(values.shape[1]), table.columns)
    ax.set_yticks(range(values.shape[0]), table.index)
    ax.tick_params(length=0)
    return image


# ---------------------------- 그래프 ----------------------------
def fig_class_distribution():
    """그림 2: train 레이블의 클래스 비율."""
    y = pd.read_csv(KAGGLE / 'train_data.csv')['Cover_Type']
    ratio = y.value_counts(normalize=True).sort_index() * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.bar(ratio.index.astype(str), ratio.values, width=0.55, color=BLUE)
    label_bars(ax, bars, lambda v: f'{v:.1f}%')
    ax.set_xlabel('클래스 (Cover_Type)')
    ax.set_ylabel('train 레이블 비율 (%)')
    ax.set_ylim(0, ratio.max() * 1.18)
    ax.grid(axis='x', visible=False)
    save(fig, 'fig02_class_distribution.png')


def fig_progress():
    """그림 1: 단계별 최고 CV accuracy (seed 42, 5-fold)."""
    stage3 = pd.read_csv(RESULTS / 'stage3' / 'final_comparison.csv')
    stage45 = pd.read_csv(RESULTS / 'stage4_5' / 'final_comparison.csv')
    points = [
        ('1단계\n넓은 탐색', pd.read_csv(RESULTS / 'stage1' / 'best_per_model.csv').cv_acc_mean.max()),
        ('2단계\n좁은 탐색', pd.read_csv(RESULTS / 'stage2' / 'best_per_model.csv').cv_acc_mean.max()),
        ('3단계\n앙상블', stage3.acc_seed42.max()),
        ('4단계\nfeature', pd.read_csv(RESULTS / 'stage4' / 'best_per_config.csv').cv_acc_mean.max()),
        ('4.5단계\n노이즈 제거', stage45.acc_seed42.max()),
    ]
    names, values = zip(*points)
    x = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.plot(x, values, color=BLUE, zorder=2)
    ax.scatter(x, values, s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for xi, v in zip(x, values):
        ax.annotate(f'{v:.4f}', (xi, v), xytext=(0, 8), textcoords='offset points',
                    ha='center', color=TEXT, fontsize=9)
    ax.axvspan(-0.4, 2.4, color='#f3f2ef', zorder=0)
    ax.text(1, 0.5845, f'모델 튜닝 / 앙상블\n{values[2] - values[0]:+.4f}', ha='center', color=TEXT_2, fontsize=9)
    ax.text(3.5, 0.5845, f'feature 추가 / 노이즈 제거\n{values[4] - values[2]:+.4f}', ha='center', color=TEXT_2, fontsize=9)
    ax.set_xticks(x, names)
    ax.set_xlim(-0.4, len(values) - 0.6)
    ax.set_ylim(0.570, 0.610)
    ax.set_ylabel('CV accuracy (seed 42, 5-fold)')
    ax.grid(axis='x', visible=False)
    save(fig, 'fig01_progress.png')


def fig_dt_depth():
    """그림 3: Decision Tree의 max_depth에 따른 train / CV accuracy (과적합)."""
    r = pd.read_csv(RESULTS / 'stage1' / 'results.csv')
    dt = r[(r.model == 'DecisionTree') & (r.min_samples_leaf == 1) & (r.criterion == 'gini')
           & r.class_weight.isna()].copy()
    dt['depth'] = dt.max_depth.fillna(999)
    dt = dt.sort_values('depth')
    labels = [str(int(d)) if d < 999 else '제한 없음\n(Baseline)' for d in dt.depth]
    x = np.arange(len(dt))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    line(ax, x, dt.train_acc_mean, ORANGE, 'train accuracy (외운 데이터)')
    line(ax, x, dt.cv_acc_mean, BLUE, 'CV accuracy (처음 보는 데이터)')
    best = dt.cv_acc_mean.to_numpy().argmax()
    ax.annotate(f'CV 최고 {dt.cv_acc_mean.iloc[best]:.3f}', (best, dt.cv_acc_mean.iloc[best]),
                xytext=(0, -18), textcoords='offset points', ha='center', color=TEXT, fontsize=9)
    ax.annotate(f'train {dt.train_acc_mean.iloc[-1]:.2f} / CV {dt.cv_acc_mean.iloc[-1]:.3f}',
                (x[-1], dt.cv_acc_mean.iloc[-1]), xytext=(-6, -16), textcoords='offset points',
                ha='right', color=TEXT, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_xlabel('max_depth (트리 최대 깊이)')
    ax.set_ylabel('accuracy')
    ax.set_ylim(0.3, 1.05)
    ax.legend(loc='upper left')
    save(fig, 'fig03_dt_depth.png')


def fig_stage1_models():
    """그림 4: 1단계 모델별 최고 CV accuracy (단일 모델 vs 앙상블)."""
    best = pd.read_csv(RESULTS / 'stage1' / 'best_per_model.csv').sort_values('cv_acc_mean')
    names = {'Baseline': 'Baseline (DT 기본값)', 'DecisionTree': 'Decision Tree',
             'AdaBoost': 'AdaBoost', 'HistGradientBoosting': 'HistGradientBoosting',
             'Bagging': 'Bagging', 'RandomForest': 'Random Forest'}
    single = best.model.isin(['Baseline', 'DecisionTree'])

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.barh([names[m] for m in best.model], best.cv_acc_mean, height=0.55,
                   color=np.where(single, MUTED, BLUE))
    label_bars(ax, bars, lambda v: f'{v:.4f}', horizontal=True)
    ax.set_xlim(0, 0.68)
    ax.set_xlabel('CV accuracy (각 모델의 가장 좋은 설정)')
    ax.grid(axis='y', visible=False)
    ax.barh([0], [0], color=MUTED, label='단일 트리')
    ax.barh([0], [0], color=BLUE, label='앙상블 (트리 여러 개)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, 'fig04_stage1_models.png')


def fig_rf_heatmap():
    """그림 5: 2단계 Random Forest 탐색 (max_features x min_samples_leaf)."""
    r = pd.read_csv(RESULTS / 'stage2' / 'results.csv')
    rf = r[(r.model == 'RandomForest') & (r.n_estimators == 300) & r.max_samples.isna()
           & (r.criterion == 'entropy')]
    table = rf.pivot_table(index='max_features', columns='min_samples_leaf', values='cv_acc_mean')
    order = ['0.1', 'sqrt', '0.2', '0.3', '0.4', '0.5']
    table.index = table.index.astype(str)
    table = table.loc[[o for o in order if o in table.index]]
    table.columns = [int(c) for c in table.columns]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    image = heatmap(ax, table, lambda v: f'{v:.4f}')
    ax.set_xlabel('min_samples_leaf (잎 하나의 최소 샘플 수)')
    ax.set_ylabel('max_features (split마다 보는 feature 비율)')
    fig.colorbar(image, ax=ax, shrink=0.85).outline.set_visible(False)
    save(fig, 'fig05_rf_heatmap.png')


def fig_agreement():
    """그림 6: 3단계 모델끼리 예측이 같은 비율."""
    table = pd.read_csv(RESULTS / 'stage3' / 'agreement_seed42.csv', index_col=0) * 100
    short = {'RandomForest': 'RF', 'Bagging': 'Bagging', 'HistGradientBoosting': 'HGB', 'AdaBoost': 'AdaBoost'}
    table = table.rename(index=short, columns=short)
    values = table.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)        # 자기 자신과의 비교(100%)는 빼고 색칠
    masked = pd.DataFrame(values, index=table.index, columns=table.columns)

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    image = heatmap(ax, masked, lambda v: '' if np.isnan(v) else f'{v:.1f}%')
    for i in range(len(table)):
        ax.text(i, i, '-', ha='center', va='center', color=TEXT_2)
    fig.colorbar(image, ax=ax, shrink=0.85).outline.set_visible(False)
    save(fig, 'fig06_agreement.png')


def fig_feature_sets():
    """그림 7: 4단계 feature 세트 x 모델 (최고 CV accuracy)."""
    best = pd.read_csv(RESULTS / 'stage4' / 'best_per_config.csv')
    sets = {'A_original': 'A 원래 그대로 (54개)', 'B_compress': 'B one-hot 압축 (12개)',
            'C_fe': 'C 새 feature 추가 (66개)', 'D_compress_fe': 'D 압축 + 새 feature (24개)'}
    y = {s: i for i, s in enumerate(reversed(list(sets)))}

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for model, color in MODEL_COLOR.items():
        rows = best[best.model == model]
        ax.scatter(rows.cv_acc_mean, [y[s] for s in rows.features], s=70, color=color,
                   edgecolor=SURFACE, linewidth=1.5, label=model, zorder=3)
    for s in sets:
        rows = best[best.features == s]
        ax.plot([rows.cv_acc_mean.min(), rows.cv_acc_mean.max()], [y[s], y[s]],
                color=GRID, linewidth=2, zorder=1)
    top = best.loc[best.cv_acc_mean.idxmax()]
    ax.annotate(f'{top.cv_acc_mean:.4f}', (top.cv_acc_mean, y[top.features]), xytext=(8, 0),
                textcoords='offset points', va='center', color=TEXT, fontsize=9)
    ax.set_yticks(list(y.values()), [sets[s] for s in y])
    ax.set_xlabel('CV accuracy (각 조합의 가장 좋은 설정)')
    ax.set_xlim(0.565, 0.607)
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, 'fig07_feature_sets.png')


def fig_importance():
    """그림 8: 4단계 최고 모델의 feature 중요도 상위 15개."""
    text = (RESULTS / 'stage4' / 'best_setting.txt').read_text(encoding='utf-8')
    lines = text.split('feature 중요도 상위 15개:\n')[1].strip().splitlines()
    names = [ln.split()[0] for ln in lines]
    values = [float(ln.split()[1]) for ln in lines]
    original = set(pd.read_csv(KAGGLE / 'test_data.csv', nrows=1).columns)
    colors = [MUTED if n in original else BLUE for n in names]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(names[::-1], values[::-1], height=0.6, color=colors[::-1])
    ax.set_xlabel('feature 중요도 (ExtraTrees, 합 = 1)')
    ax.grid(axis='y', visible=False)
    ax.barh([names[-1]], [0], color=BLUE, label='새로 만든 feature')
    ax.barh([names[-1]], [0], color=MUTED, label='원래 feature')
    ax.legend(loc='lower right')
    save(fig, 'fig08_importance.png')


def fig_et_leaf():
    """그림 9: ExtraTrees(max_features 0.7)의 min_samples_leaf에 따른 train / CV accuracy."""
    r4 = pd.read_csv(RESULTS / 'stage4' / 'results.csv')
    r45 = pd.read_csv(RESULTS / 'stage4_5' / 'results.csv')
    a = r4[(r4.model == 'ExtraTrees') & (r4.features == 'C_fe') & (r4.max_features == 0.7)]
    b = r45[(r45.model == 'ExtraTrees') & (r45.n_estimators == 300) & (r45.max_features == 0.7)]
    df = pd.concat([a, b]).drop_duplicates('min_samples_leaf').sort_values('min_samples_leaf')
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    line(ax, x, df.train_acc_mean, ORANGE, 'train accuracy')
    line(ax, x, df.cv_acc_mean, BLUE, 'CV accuracy')
    best = df.cv_acc_mean.to_numpy().argmax()
    ax.annotate(f'CV 최고 {df.cv_acc_mean.iloc[best]:.4f}', (best, df.cv_acc_mean.iloc[best]),
                xytext=(0, -18), textcoords='offset points', ha='center', color=TEXT, fontsize=9)
    ax.set_xticks(x, df.min_samples_leaf.astype(int))
    ax.set_xlabel('min_samples_leaf (클수록 단순한 트리)')
    ax.set_ylabel('accuracy')
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='upper right')
    save(fig, 'fig09_et_leaf.png')


def fig_noise_removal():
    """그림 10: 4.5단계 노이즈 레이블 제거 비율에 따른 CV accuracy."""
    r = pd.read_csv(RESULTS / 'stage4_5' / 'results.csv')
    f = r[r.model == 'ExtraTrees+filter']
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for leaf, color in zip(sorted(f.min_samples_leaf.unique()), [BLUE, ORANGE]):
        rows = f[f.min_samples_leaf == leaf].sort_values('remove_fraction')
        line(ax, rows.remove_fraction * 100, rows.cv_acc_mean, color, f'min_samples_leaf = {int(leaf)}')
    top = f.loc[f.cv_acc_mean.idxmax()]
    ax.annotate(f'최고 {top.cv_acc_mean:.4f}', (top.remove_fraction * 100, top.cv_acc_mean),
                xytext=(0, 9), textcoords='offset points', ha='center', color=TEXT, fontsize=9)
    ax.set_xlabel('제거한 train 샘플 비율 (%)  -  0% = 제거 안 함')
    ax.set_ylabel('CV accuracy (seed 42)')
    ax.set_xticks(sorted(f.remove_fraction.unique() * 100))
    ax.legend(loc='lower right')
    save(fig, 'fig10_noise_removal.png')


def fig_removed_by_class():
    """그림 11: 25% 제거할 때 각 클래스 레이블이 제거된 비율."""
    table = pd.read_csv(RESULTS / 'stage4_5' / 'removed_by_class.csv')
    row = table.iloc[(table.remove_fraction - 0.25).abs().argmin()]
    classes = [c for c in table.columns if c.startswith('class_')]
    values = [row[c] * 100 for c in classes]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.bar([c.split('_')[1] for c in classes], values, width=0.55, color=BLUE)
    label_bars(ax, bars, lambda v: f'{v:.0f}%')
    ax.set_xlabel('클래스 (레이블 기준)')
    ax.set_ylabel('그 클래스 레이블 중 제거된 비율 (%)')
    ax.set_ylim(0, 100)
    ax.grid(axis='x', visible=False)
    save(fig, 'fig11_removed_by_class.png')


def fig_label_vs_pred():
    """그림 13: train 레이블 비율 vs 최종 모델의 test 예측 비율."""
    y = pd.read_csv(KAGGLE / 'train_data.csv')['Cover_Type']
    pred = pd.read_csv(RESULTS / 'stage4_5' / 'submission_stage4_5_best.csv')['Cover_Type']
    classes = np.sort(y.unique())
    train_ratio = [(y == c).mean() * 100 for c in classes]
    pred_ratio = [(pred == c).mean() * 100 for c in classes]
    x = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    ax.bar(x - 0.15, train_ratio, width=0.28, color=MUTED, label='train 레이블 (노이즈 포함)')
    ax.bar(x + 0.15, pred_ratio, width=0.28, color=BLUE, label='최종 모델의 test 예측')
    for xi, (a, b) in enumerate(zip(train_ratio, pred_ratio)):
        if classes[xi] in (4, 5):
            ax.annotate(f'{a:.1f}% → {b:.1f}%', (xi, max(a, b)), xytext=(0, 6),
                        textcoords='offset points', ha='center', color=TEXT, fontsize=9)
    ax.set_xticks(x, classes)
    ax.set_xlabel('클래스')
    ax.set_ylabel('비율 (%)')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper right')
    save(fig, 'fig13_label_vs_pred.png')


def fig_final_comparison():
    """그림 12: 최종 비교 (seed 3개 x 5 fold = 15개 점수의 평균 ± 표준편차)."""
    table = pd.read_csv(RESULTS / 'stage4_5' / 'final_comparison.csv')
    table = table[~table.method.str.startswith('Baseline')].sort_values('cv_acc_mean')
    rename = {
        'DecisionTree (A_original, 3단계)': 'Decision Tree (1단계 설정)',
        'RandomForest (A_original, 3단계)': 'Random Forest (2단계)',
        'Voting: RF+Bag (A_original, 3단계)': 'Voting: RF+Bagging (3단계)',
        'Stacking: RF+Bag+HGB+Ada (A_original, 3단계)': 'Stacking 4개 모델 (3단계)',
        'RF (C_fe, 4단계)': 'Random Forest + 새 feature (4단계)',
        'ET (C_fe, 4단계)': 'ExtraTrees + 새 feature (4단계)',
        'ET 범위 확장 (4.5단계 a)': 'ExtraTrees 범위 확장 (4.5단계 a)',
    }
    names = [rename.get(m, m.replace('ET + 노이즈', 'ExtraTrees + 노이즈').replace(' (4.5단계 b)', ' (4.5단계 b, 최종)'))
             for m in table.method]
    final = np.arange(len(table)) == len(table) - 1
    y = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.errorbar(table.cv_acc_mean, y, xerr=table.cv_acc_std, fmt='none', ecolor='#d6d4cf', elinewidth=2, zorder=1)
    ax.scatter(table.cv_acc_mean, y, s=60, color=np.where(final, BLUE, MUTED),
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for yi, (v, is_final) in enumerate(zip(table.cv_acc_mean, final)):
        ax.annotate(f'{v:.4f}', (v, yi), xytext=(0, 7), textcoords='offset points', ha='center',
                    color=TEXT if is_final else TEXT_2, fontsize=8.5)
    ax.set_yticks(y, names)
    ax.set_xlabel('CV accuracy (seed 3개 x 5 fold = 15개 점수, 막대 = ± 표준편차)')
    ax.grid(axis='y', visible=False)
    save(fig, 'fig12_final_comparison.png')


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig_class_distribution()
    fig_progress()
    fig_dt_depth()
    fig_stage1_models()
    fig_rf_heatmap()
    fig_agreement()
    fig_feature_sets()
    fig_importance()
    fig_et_leaf()
    fig_noise_removal()
    fig_removed_by_class()
    fig_label_vs_pred()
    fig_final_comparison()


if __name__ == '__main__':
    main()
