"""
Created on Thu Jun 23 02:51:09 2022
This script helps to explore the dataset by printing the more information
@author Pavan Muguda Sanjeevamurthy
"""

import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)
import logging
from termcolor import cprint


class PrintInfo:
    """
    Class of functions that are helpful for exploring the dataset
    """
    def __init__(self, dataframe):
        """
        Initializes the dataframe to the class
        :param dataframe: dataframe whose data tobe explored
        """
        self.data = dataframe
        self.keys = list(self.data.columns)
        
    def sneak_peek(self, train=True):
        """
        This function gives a initial impressions of the dataset
        :param train: bool helpful for printing statements
        :return: pandas dataframe containing first five rows
        """

        if train:
            cprint('First sneak peak into the train data', color='green')
        else:
            cprint('First sneak peak into the test data', color='green')
        cprint('------------------------------------------------------', color='red')
        first_five_rows = pd.DataFrame(self.data.head(5))
        num_of_customers, num_of_features = self.data.shape
        cprint(f'1. Number of records in the data are {num_of_customers}', color='grey')
        cprint(f'2. Number of feature vectors in the data are {num_of_features}', color='grey')
        cprint(f'3. The first 5 records of the data', color='grey')
        return first_five_rows

    def print_all_unique_elements(self, train =True):
        """
        This function returns all the unique values of categorical variables
        :param train: bool helpful for printing statements
        :return: pandas dataframe containing categorical features and its respective unique labels
        """
        if train:
            cprint('unique elements in categorical feature of the train data:', color='green')
        else:
            cprint('unique elements in categorical feature of the test data:', color='green')
        cprint('-----------------------------------------------------------------------------', color='red')
        unique_elements_list = []
        for key in self.keys:
            if self.data[key].dtype == 'object':
                unique_elements = self.data[key].unique()
                if len(unique_elements) < 15:
                    temp = [key, list(unique_elements)]
                    unique_elements_list.append(temp)
        return pd.DataFrame(unique_elements_list, columns=['Categorical features', 'Unique elements                            '])

    def no_entry_values_in_features(self, train=True):
        """
        This function returns count of all the features which have NaN values
        :param train: bool helpful for printing statements
        :return:
        """

        if train:
            cprint('Number of NaN values in features of the train data', color='green')
        else:
            cprint('Number of NaN values in features of the test data', color='green')
        cprint('-----------------------------------------------------', color='red')
        nan_elements_list = []
        for key in self.keys:
            is_nan_elements = self.data[key].isna().sum()
            if is_nan_elements > 0:
                # print(f'{key} has {is_nan_elements} NaN elements')
                missing_perc = (self.data[key].isnull().sum() / self.data[key].isnull().count())*100
                temp = [key, is_nan_elements, missing_perc]
                nan_elements_list.append(temp)
        return pd.DataFrame(nan_elements_list, columns=['Features with NaN', 'Num of NaNs', 'NaN elements (%)'])
                
    def visualize_feature_plots(self):
        """
        This function plots all the features of the data
        :return: None
        """
        features_with_numbers = self.data.select_dtypes(include='number')
        features_with_objects = self.data.select_dtypes(include='object')
        # plotting numerical features as a histogram
        print(' Numerical Features ')
        features_with_numbers.hist(bins=60, figsize = (20, 10))
        # plotting categorical features as a bar chart
        print(' Categorical Features ')
        categorical_features = ['PARENT1', 'MSTATUS', 'EDUCATION', 'JOB', 'CAR_USE', 'CAR_TYPE', 'RED_CAR', 'REVOKED','URBANICITY']
        num_rows = 3; num_cols = 3
        gridspec_kw = dict(left=0.02, right=0.92, top=1 - 0.02, bottom=0.02, hspace=0.98, wspace=0.30)
        fig, ax = plt.subplots(num_rows, num_cols, figsize=[16, 9], gridspec_kw=gridspec_kw)
        fig_index = 0
        for row in range(num_rows):
            for col in range(num_cols):
                features_with_objects[categorical_features[fig_index]].value_counts().plot(kind='bar', ax=ax[row, col]).set_title(
                    categorical_features[fig_index])
                fig_index += 1
