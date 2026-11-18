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


            raise NotImplementedError("Complete EDIT HERE 1")
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


        raise NotImplementedError("Complete EDIT HERE 2")
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


            raise NotImplementedError("Complete EDIT HERE 3")
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


            raise NotImplementedError("Complete EDIT HERE 4")
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


        raise NotImplementedError("Complete EDIT HERE 5")
        # ==================== EDIT HERE 5: END ======================

        return prediction
