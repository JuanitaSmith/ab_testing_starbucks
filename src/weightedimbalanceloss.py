""" Class to use calculate Weighted Imbalance Loss to be used as XGBOOST custom objective function """

import numpy as np

class WeightedImbalanceLoss:
    """
    Class to use calculate Weighted Imbalance Loss to be used as XGBOOST custom objective function.
    Inspired by: https://github.com/handongfeng/Xgboost-With-Imbalance-And-Focal-Loss/tree/master

    This function add more weight when label 1 is misclassified

    Args:
        imbalance_alpha: A value greater than 1 means to put extra loss on 'classifying 1 as 0'.
        pred: untransformed leaf weight when custom objective is provided.
        dtrain: training data in xgboost dmatrix format
    """

    def __init__(self, imbalance_alpha):
        """
        :param imbalance_alpha: the imbalanced \alpha value for the minority class (label as '1')
        """
        self.imbalance_alpha = imbalance_alpha

    def weighted_binary_cross_entropy(self, pred, dtrain):
        # assign the value of imbalanced alpha
        imbalance_alpha = self.imbalance_alpha
        # retrieve data from dtrain matrix
        label = dtrain.get_label()

        # Convert raw leaf weights to probabilities
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))

        grad = -(imbalance_alpha ** label) * (label - sigmoid_pred)
        hess = (imbalance_alpha ** label) * sigmoid_pred * (1.0 - sigmoid_pred)

        return grad, hess
