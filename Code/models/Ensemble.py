import numpy as np
from sklearn.tree import DecisionTreeClassifier


"""
Bagging / AdaBoost implementation.

[Data]
    X : input features, shape (N, D)
    y : labels (-1 or +1), shape (N,)

[Student implementation]
    (a) Bagging  : EDIT HERE 1, 2
    (b) AdaBoost: EDIT HERE 3, 4, 5
    Do not modify code outside these blocks.

Student template: complete the five EDIT HERE blocks below.
"""


class Bagging:
    def __init__(self, n_estimators=50, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        self.bootstrap_indices_ = []

    def fit(self, X, y):
        """
        Train each tree using a bootstrap sample.

        [Inputs]
            X : training features, shape (N, D)
            y : training labels, shape (N,)
        [Output]
            self : fitted Bagging model
        """
        num_samples = len(X)
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.bootstrap_indices_ = []

        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state + i,
            )

            indices = None
            # ==================== EDIT HERE 1: START ====================
            # 1. Sample N indices WITH replacement.
            # 2. Train tree using the sampled X and y.
            # Write your code here.

            # num_samples = N, 학습 data 개수
            # rng = random number generator
            # tree = 이번에 학습할 tree
            # indices = 뽑힌 행 번호 N개짜리 배열이고, 아래에서 bootstrap_indices_에 저장

            indices = rng.choice(num_samples, size=num_samples, replace=True)
            tree.fit(X[indices], y[indices])
            # ==================== EDIT HERE 1: END ======================

            self.estimators_.append(tree)
            self.bootstrap_indices_.append(indices)

        return self

    def predict(self, X):
        """
        Predict by majority vote. A tied vote returns +1.

        [Input]
            X : input features, shape (N, D)
        [Output]
            prediction : predicted labels (-1 or +1), shape (N,)
        """
        num_samples = len(X)
        vote_sum = np.zeros(num_samples)
        prediction = None

        # ==================== EDIT HERE 2: START ====================
        # 1. Add the predictions (-1/+1) of all trained trees.
        # 2. Return +1 when vote_sum >= 0, otherwise return -1.
        # Write your code here.

        
        # self.estimators_ = 학습된 트리 리스트 
        # vote_sum = 샘플마다 투표 합을 담는 배열이에요. 0으로 초기화돼 있어요
        # prediction = 최종 예측 (−1/+1) 배열

        for tree in self.estimators_ :
            vote_sum += tree.predict(X)
        prediction = np.where(vote_sum >= 0, 1, -1)

        # ==================== EDIT HERE 2: END ======================

        return prediction


class AdaBoost:
    def __init__(self, n_estimators=50, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        self.alphas_ = []
        self.errors_ = []
        self.weight_history_ = []

    def fit(self, X, y):
        """
        Train decision stumps sequentially using sample weights.

        [Inputs]
            X : training features, shape (N, D)
            y : training labels (-1 or +1), shape (N,)
        [Output]
            self : fitted AdaBoost model
        """
        num_samples = len(X)
        weights = np.ones(num_samples) / num_samples
        self.estimators_ = []
        self.alphas_ = []
        self.errors_ = []
        self.weight_history_ = [weights.copy()]
        self.fallback_label_ = 1 if np.sum(y) >= 0 else -1

        for i in range(self.n_estimators):
            # Provided: fit a depth-1 tree with CURRENT sample weights.
            tree = DecisionTreeClassifier(
                max_depth=1,
                random_state=self.random_state + i,
            )
            tree.fit(X, y, sample_weight=weights)
            prediction = tree.predict(X)

            error = None
            # ==================== EDIT HERE 3: START ====================
            # Sum the weights of incorrectly classified samples.
            # Write your code here.

            # 가중 오류율: ε = Σ wᵢ · I(h(xᵢ) ≠ yᵢ). 여기서 I(...)는 조건이 참이면 1, 거짓이면 0
            # 모델 가중치: α = ½ · ln((1 − ε) / ε)
            # 샘플 가중치 갱신: wᵢ ← wᵢ · exp(−α · yᵢ · h(xᵢ)), 그다음 합이 1이 되도록 정규화

            # 1. weights = 샘플 가중치 (N,). 처음엔 모두 1/N이고, 합은 항상 1
            # 2. prediction = 이번 stump에서 학습 data에 낸 예측 (-1/+1)
            # 3. error = weighted error rate. 가중 오류율 ε
            # 4. safe_error = error를 아주 작은 값 ~ 1 사이로 잘라 둔 값. log(0)이나 0으로 나누기를 막아 줌.
            # 5. alpha = 이 stump의 신뢰도

            mask = (prediction != y)
            error = np.sum(weights * mask)
            # ==================== EDIT HERE 3: END ======================

            # Provided: reject a learner no better than random guessing.
            if error >= 0.5 - 1e-12:
                break

            # Provided: avoid division by zero and log(0).
            safe_error = np.clip(error, 1e-12, 1 - 1e-12)

            alpha = None
            # ==================== EDIT HERE 4: START ====================
            # 1. Compute alpha using safe_error and the formula in HW4.
            # 2. Multiply weights by exp(-alpha * y * prediction).
            # 3. Normalize weights so their sum is 1.
            # Write your code here.

            # 모델 가중치: α = ½ · ln((1 − ε) / ε)

            alpha = 0.5 * np.log((1 - safe_error) / safe_error)
            weights *= np.exp(-alpha * y * prediction)
            weights = weights / np.sum(weights) # normalization!

            # ==================== EDIT HERE 4: END ======================

            self.estimators_.append(tree)
            self.alphas_.append(alpha)
            self.errors_.append(error)
            self.weight_history_.append(weights.copy())

            # Provided: stop after a perfect classifier.
            if error <= 1e-12:
                break

        return self

    def predict(self, X):
        """
        Predict by weighted vote. A tied vote returns +1.

        [Input]
            X : input features, shape (N, D)
        [Output]
            prediction : predicted labels (-1 or +1), shape (N,)
        """
        num_samples = len(X)

        # Provided: use the majority class if no useful tree was found.
        if len(self.estimators_) == 0:
            return np.full(num_samples, self.fallback_label_)

        score = np.zeros(num_samples)
        prediction = None

        # ==================== EDIT HERE 5: START ====================
        # 1. Add alpha * tree_prediction for each trained tree.
        # 2. Return +1 when score >= 0, otherwise return -1.
        # Write your code here.

        # 1. self.estimators_ = 학습된 stump
        # 2. self.alphas_ = 각 stump의 신뢰도
        # 3. score = 샘플마다 가중 투표 합을 담는 배열. 0으로 초기화 되어있음

        for tree, alpha in zip(self.estimators_, self.alphas_):
            score += alpha * tree.predict(X)
        prediction = np.where(score >= 0, 1, -1)

        # ==================== EDIT HERE 5: END ======================

        return prediction
