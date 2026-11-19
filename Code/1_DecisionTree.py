import os.path
import numpy as np
from models.DecisionTree import DecisionTree
from utils import *
import random
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(1)
random.seed(1)


def main():
    data_name = 'TicTacToe'
    # =============================== EDIT HERE ===============================
    max_tree_depth = 6
    # =========================================================================
    
    print(f"> data_name: {data_name}")
    print(f"> tree_depth: {max_tree_depth}")
    
    data_df = pd.read_csv(os.path.join('data/tictactoe', 'tictactoe.csv'))
    train_df, test_df = train_test_split(data_df, test_size=0.2, stratify=data_df['V10'], random_state=42)

    x_test, y_test = split_to_x_y(test_df, test_df.columns.values)

    Dtree = DecisionTree(max_depth=max_tree_depth)
    Dtree.fit(train_df, data_name)

    pred_test = Dtree.predict(x_test)
    y_test = y_test.values
    test_accuracy = np.sum((pred_test == y_test)) / len(y_test)
    print('Test accuracy: %.4f' % test_accuracy)


if __name__ == '__main__':
    main()
