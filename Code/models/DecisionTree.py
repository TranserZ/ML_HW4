import numpy as np


def compute_entropy(labels):
    entropy = None
    # =============================== EDIT HERE ===============================
    """ 
    Compute the entropy of the labels.
    The entropy is defined as:
    H(X) = -sum(p(x) * log2(p(x))) for all x in X
    where p(x) is the probability of x in the labels.
    """

    # labels = 정답 array. ['positive', 'negative', 'positive', ...] 같은 문자열 1차원 배열
    
    # 1. class별 개수 count
    # 2. count를 확률로 바꾼다. 공식: p(x) = count(x) / total_count
    # 3. 공식 -Σ p·log₂(p)를 적용해서 entropy를 계산한다.

    distinct_values, count = np.unique(labels, return_counts=True)
    total_count = len(labels)
    prob = count / total_count
    entropy = -np.sum(prob * np.log2(prob))
    # =========================================================================

    return entropy


class Node():
    def __init__(self, dataFrame, depth, split_feature=''):
        self.df = dataFrame
        self.depth = depth
        self.split_feature = split_feature

        self.entropy = compute_entropy(self.df.values[:, -1])
        self.child = {}
        self.is_leaf = False
        self.label = None


class DecisionTree():
    def __init__(self, max_depth, feature_select=False, num_features=3, verbose=0):
        self.max_depth = max_depth
        self.threshold = 0
        self.verbose = verbose
        self.feature_select = feature_select
        self.num_features = num_features

    def fit(self, train_df, data_name):
        self.root = Node(train_df, 0)

        self.branch_name = '(Root)'
        self.data_name = data_name
        
        self.generate_tree(self.root)
    
    def is_reached_max_depth(self, node: Node):
        if node.depth == self.max_depth:
            return True
        else:
            return False

    def is_reached_single_class(self, node: Node):
        if node.entropy == 0:
            return True
        else:
            return False

    def is_reached_stopping_criteria(self, node):
        """
        (when tree_depth == maximum_depth or impurity == 0), Set current node to leaf_node
        and decide class label.
        """
        if self.is_reached_max_depth(node) or self.is_reached_single_class(node):
            node.is_leaf = True
            labels = node.df.values[:, -1]
            node.label = 'positive' if (np.sum(labels == 'positive') > np.sum(labels == 'negative')) else 'negative'

            if self.verbose:
                print('\t'*node.depth, 'p:', np.sum(labels == 'positive'), 'n:', np.sum(labels == 'negative'))
                print('\t'*node.depth, 'Parent:', self.branch_name)
                print('\t'*node.depth, f'\__Depth \033[32m{node.depth}\033[0m | \033[95mLeaf node\033[0m | \033[95mLabel: {node.label}\033[0m')

            return True

        return False

    def selection_criteria(self, df):
        if self.feature_select:
            data_df = self.random_feature_selection(df, self.num_features)
        else:
            data_df = df

        input_features = data_df.columns.values[:-1]
        labels = data_df.values[:, -1]

        # 특성 다양성이 없으면 분할 중지
        if len(input_features) == 0 or df.shape[0] <= 1:
            return None

        max_gain = -np.inf
        best_feature = None

        for feature in input_features:
            feature_data = data_df[feature].values
            
            # # categorical features
            distinct_values, counts = np.unique(feature_data, return_counts=True)
            entropy = None
            conditional_entropy = None
            information_gain = None
            
            # =============================== EDIT HERE ===============================
               
            """
            Compute the information gain of the feature.
            The information gain is defined as:
            IG(X|Y) = H(X) - H(X|Y)
            where H(X) is the entropy of the labels and H(X|Y) is the conditional entropy of the labels given the feature Y.
            The conditional entropy is defined as:
            H(X|Y) = sum(p(y) * H(X|Y=y)) for all y in Y
            """

            # Given: 
            # labels: 현재 노드의 정답 전체
            # feature_data: 지금 보고 있는 feature 열의 값 (예: x, o, b)
            # distinct_values, counts: 그 feature가 가지는 값의 종류와 각각의 개수

            # Not Given:
            # 1. entropy, H(x): 현재 노드 정답 전체의 엔트로피
            # 2. conditional_entropy, H(x|y): 
            #   p(y) = 그 값의 개수 ÷ 전체 개수
            #   H(X|Y=y) = feature가 y인 행의 labels만 골라서 구한 엔트로피
            # 3. information_gain, IG(x|y): H(x) - H(x|Y)

            # 1. entropy, H(x)
            entropy = compute_entropy(labels)
            # 2. conditional_entropy, H(x|y)
            conditional_entropy = 0
            for value, count in zip(distinct_values, counts):
                p_y = count / len(feature_data)
                entropy_y = compute_entropy(labels[feature_data == value])
                conditional_entropy += p_y * entropy_y
            # 3. information_gain, IG(x|y)
            information_gain = entropy - conditional_entropy
            # =========================================================================

            if information_gain > max_gain:
                max_gain = information_gain
                best_feature = feature
        if max_gain <= 0:
            return None
        return best_feature

    def random_feature_selection(self, df, num_features):
        feature_list = np.array(df.columns[:-1])
        label_col = np.array(df.columns[-1])
        
        np.random.shuffle(feature_list)
        random_features = feature_list[:num_features]

        return df[np.append(random_features, label_col)]

    def generate_tree(self, node: Node):
        df = node.df

        if self.is_reached_stopping_criteria(node):
            return

        best_feature = self.selection_criteria(df)
        
        if best_feature is None:
            node.is_leaf = True
            labels = node.df.values[:, -1]
            node.label = 'positive' if (np.sum(labels == 'positive') > np.sum(labels == 'negative')) else 'negative'
            return
            
        node.split_feature = best_feature
        feature_data = df[best_feature]

        if self.verbose:
            print('\t'*node.depth, 'Parent:', self.branch_name)

        distinct_values = np.unique(feature_data)

        for i, value in enumerate(distinct_values):
            if self.verbose:
                print('\t'*node.depth, f'\__Depth \033[32m{node.depth}\033[0m | Best Feature: \033[32m{node.split_feature}\033[0m | {i}th branch: \033[32m{value}\033[0m')
            
            child_df = df[feature_data == value].copy()
            
            if child_df.shape[0] == 0:
                continue
                
            child_node = Node(child_df, node.depth + 1, '%s = %s' % (best_feature, value))
            node.child[value] = child_node

            self.branch_name = f'({best_feature}/{str(value)})'
            self.generate_tree(node.child[value])

    def predict(self, x):
        pred = np.full(len(x), None)

        for i, idx in enumerate(x.index):
            cur_node = self.root
            x_pred = x.loc[idx]
            
            while True:
                if cur_node.is_leaf:
                    pred[i] = cur_node.label
                    break

                value = x_pred[cur_node.split_feature]

                if value not in cur_node.child:
                    labels = cur_node.df.values[:, -1]
                    pred[i] = 'positive' if np.sum(labels == 'positive') > np.sum(labels == 'negative') else 'negative'
                    break
                cur_node = cur_node.child[value]

        return pred

