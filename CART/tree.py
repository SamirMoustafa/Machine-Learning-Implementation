import numpy as np


class decision_tree(object):
    MAX_VALUE = 999

    def __init__(self, max_depth, min_num_split=2):
        self.max_depth = max_depth
        self.min_num_sample = min_num_split

    def gini_index(self, features, classes):
        """
        :param features: Features in data
        :param classes: Classes in every feature
        :return: 1 - summation{p^2 * deviation of each group}
        """
        n_samples = sum([len(group) for group in features])
        gini = 0
        for group in features:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0

            for class_val in classes:
                p = (group[:, -1] == class_val).sum() / size
                score += p ** 2
            gini += (1.0 - score) * (size / n_samples)
        return gini

    def split(self, feat, val, Xy):
        """
        :param feat: Feat to split about it
        :param val: Value to split about it
        :param Xy: Dataset as whole
        :return: Two splits dataset for left and right node
        """
        Xi_left = np.array([]).reshape(0, self.Xy.shape[1])
        Xi_right = np.array([]).reshape(0, self.Xy.shape[1])
        for i in Xy:
            if i[feat] <= val:
                Xi_left = np.vstack((Xi_left, i))
            if i[feat] > val:
                Xi_right = np.vstack((Xi_right, i))
        return Xi_left, Xi_right

    def best_split(self, Xy):
        """
        :param Xy: Dataset
        :return: Best split based on Gini index
        """
        classes = np.unique(Xy[:, -1])
        best_feat, best_val, best_score = self.MAX_VALUE, self.MAX_VALUE, self.MAX_VALUE
        best_groups = None
        for feat in range(Xy.shape[1] - 1):
            for i in Xy:
                groups = self.split(feat, i[feat], Xy)
                gini = self.gini_index(groups, classes)

                if gini < best_score:
                    best_feat = feat
                    best_val = i[feat]
                    best_score = gini
                    best_groups = groups
        output = {}
        output['feat'] = best_feat
        output['val'] = best_val
        output['groups'] = best_groups
        return output

    def terminal_node(self, group):
        """
        :param group: Set of values to be terminal node

        Majority voting to get the terminal node
        """
        classes, counts = np.unique(group[:, -1], return_counts=True)
        return classes[np.argmax(counts)]

    def split_branch(self, node, depth):
        left_node, right_node = node['groups']
        del (node['groups'])
        if not isinstance(left_node, np.ndarray) or not isinstance(right_node, np.ndarray):
            node['left'] = node['right'] = self.terminal_node(left_node + right_node)
            return
        if depth >= self.max_depth:
            node['left'] = self.terminal_node(left_node)
            node['right'] = self.terminal_node(right_node)
            return
        if len(left_node) <= self.min_num_sample:
            node['left'] = self.terminal_node(left_node)
        else:
            node['left'] = self.best_split(left_node)
            self.split_branch(node['left'], depth + 1)
        if len(right_node) <= self.min_num_sample:
            node['right'] = self.terminal_node(right_node)
        else:
            node['right'] = self.best_split(right_node)
            self.split_branch(node['right'], depth + 1)

    def build_tree(self, Xy=None):
        """
        :param Xy: Dataset concatenated together
        :return: Tree root

        Recursively build tree, unclear if this is the correct way
        """
        if not Xy is None:
            self.Xy = Xy
        self.root = self.best_split(self.Xy)
        self.split_branch(self.root, 1)
        return self.root

    def fit(self, X, y):
        """
        :param X: Training data
        :param y: Desired output data

        Start building the tree
        """
        self.X = X
        self.y = y
        self.Xy = np.column_stack((X, y))
        self.build_tree()

    def predict_sample(self, node, sample):
        if sample[node['feat']] < node['val']:
            if isinstance(node['left'], dict):
                return self.predict_sample(node['left'], sample)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_sample(node['right'], sample)
            else:
                return node['right']

    def predict(self, X_test):
        self.y_pred = np.array([])
        for i in X_test:
            self.y_pred = np.append(self.y_pred, self.predict_sample(self.root, i))
        return self.y_pred