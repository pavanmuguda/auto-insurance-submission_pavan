# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 02:51:09 2022
This script helps cleaning the dataset.
@author: pavan
"""


from sklearn.preprocessing import MinMaxScaler
import pandas as pd
pd.set_option('display.max_columns', None)

class CleanData:
    """
    Class of functions that are used to fix dataset issues such as
    1. Missing values
    2. Inconsistent labels in cat features
    4. encoding the cat features
    3. scaling the features
    """

    def __init__(self):
        print('')

    def dollar_to_numeric(self, dataframe):
        """
        This function casts string dollar amounts columns to numerical
        :param dataframe: input dataset in pandas dataframe
        :return:None
        """
        for col in ['INCOME', 'OLDCLAIM', 'BLUEBOOK', 'HOME_VAL']:
            dataframe[col] = dataframe[col].str.replace('$', '').str.replace(',', '.')
            dataframe[col] = pd.to_numeric(dataframe[col], downcast='float')

    def perform_consistent_labelling(self, dataframe):
        """ This function correcting misspelled values
        :param dataframe: input dataset in pandas dataframe
        :return:None
        """
        dataframe['EDUCATION'] = dataframe['EDUCATION'].replace(to_replace={'<High School': 'z_High School'})

    def replace_nan_value_entries(self, dataframe):
        """
        This function deals with all the missing values in the data
        :param dataframe: input dataset in pandas dataframe
        :return: None
        """
        features_with_numbers = dataframe.select_dtypes(include='number').columns
        for fn in features_with_numbers:
            if dataframe[fn].isna().sum() > 0:
                dataframe[fn][dataframe[fn].isna()] = pd.DataFrame.mean(dataframe[fn])

        features_with_objects = dataframe.select_dtypes(include='object').columns
        for fo in features_with_objects:
            if dataframe[fo].isna().sum() > 0:
                dataframe[fo][dataframe[fo].isna()] = dataframe[fo].value_counts().index[0]

    def encoding_for_cat_features(self, dataframe):
        """
        This function does onehot encoding for cat features with two labels and ordinal encoding otherwise
        :param dataframe: input dataset in pandas dataframe
        :return: None
        """
        features_with_objects = dataframe.select_dtypes(include='object').columns
        for fo in features_with_objects:
            dataframe[fo] = dataframe[fo].astype('category').cat.codes

    def scale_feature_values(self, dataframe):
        """
        This function scales the data using MinMaxscaler
        :param dataframe: input dataset in pandas dataframe
        :return: ndarray containing scaled values
        """
        scaler = MinMaxScaler()
        scaler.fit(X=dataframe)
        scaled_array = scaler.transform(X=dataframe)
        return scaled_array

    ## Below format is followed to include CleanData in the sklearn pipeline class ##

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        This function cleans and transforms the data using above functions together
        :param X: feature vectors
        :param y: labels
        :return: transformed clean data
        """
        X_ = X.copy()
        X_ = X_.drop(['INDEX', 'TARGET_AMT', 'TARGET_FLAG'], axis=1)
        self.perform_consistent_labelling(X_)
        self.dollar_to_numeric(X_)
        self.replace_nan_value_entries(X_)
        self.encoding_for_cat_features(X_)
        scaled_X_ = self.scale_feature_values(X_)
        return pd.DataFrame(scaled_X_, columns=X_.columns)
    
