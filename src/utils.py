""" Helper functions """

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import pandas as pd


def remove_outliers(df, col):
    """ Remove records that are more than 3 standard deviations away from the mean """
    print('Shape before: {}'.format(df.shape))
    v2_mean = df[col].mean()
    v2_std = df[col].std()
    df['zscore'] = (df[col] - v2_mean) / v2_std
    nr_outliers = len(df[(df['zscore'] > 3) | (df['zscore'] < -3)])
    print('nr_outliers: {}'.format(nr_outliers))
    df = df[(df['zscore'] < 3) & (df['zscore'] > -3)]
    print('Shape after removal of outliers: {}'.format(df.shape))
    df = df.drop(['zscore'], axis=1, errors='ignore')
    return df


def feature_engineering_training(df, onehot=False, drop_first=False):
    """
    Prepare training data for machine learning when all features are present
     - Set 'ID' as the index
     - Remove outliers for V2
     - Filter data to contain only treatment group who received promotions
     - Drop columns we no longer need like 'Promotions'

     Args:
         df (pd.DataFrame): input data containing all features
         onehot (bool): Indicate of features V1, V4, V5, V6, V7 should be onehot encoded
         drop_first (bool): Drop first column during onehot-encoding to avoid relationships between independent variables
     """

    # scale, remove outliers and correct dtypes
    df = feature_engineering_testing(df, onehot=onehot, drop_first=drop_first)

    df.set_index('ID', inplace=True, verify_integrity=True)

    # df = remove_outliers(df, col='V2')

    # filter dataset to use only the treatment group which received the promotions
    df = df[(df['Promotion'] == 'Yes')]

    df.drop(['Promotion'], axis=1, inplace=True, errors='ignore')

    return df

def feature_engineering_testing(df, drop_unused=True, onehot=False, drop_first=False):
    """
    Prepare testing data for machine learning when only features V1-V7 is passed
     - Scaling: use standard scaler on V2 as data is a normal distribution
     - Set categorical features to type 'category'
     - one-hot encode if requested

     Args:
         df (pd.DataFrame): input data containing all features
         onehot (bool): Indicate of features V1, V4, V5, V6, V7 should be onehot encoded
         drop_first (bool): Drop first column during onehot-encoding to avoid relationships between independent variables
     """

    if drop_unused:
        df.drop(['V1', 'V2', 'V6', 'V7'], axis=1, inplace=True, errors='ignore')
    # df.drop(['V2'], axis=1, inplace=True, errors='ignore')

    # scaler = StandardScaler()
    # df['V2'] = scaler.fit_transform(df[['V2']])

    # categorical_cols = ['V1', 'V4', 'V5', 'V6', 'V7']
    categorical_cols = ['V4', 'V5']
    df[categorical_cols] = df[categorical_cols].astype("category")

    if onehot:
        df = pd.get_dummies(
            df,
            # columns=['V1', 'V4', 'V5', 'V6', 'V7'],
            columns=['V4', 'V5',],
            dtype='int8',
            drop_first=drop_first,
        )

    return df


def get_mi_score(X, y):
    """
    Calculate mutual information to show feature importance.

    Estimated mutual information between each feature and the target

    Args:
        X (array-like or sparse matrix) - Independent features
        y (array-like) - Dependent feature to be predicted
    Returns:
        mi (ndarray) - estimated mutual information
    """

    # mi = mutual_info_regression(X, y, random_state=10, discrete_features=True)
    mi = mutual_info_classif(X, y, random_state=10, discrete_features=True)
    mi = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return mi
