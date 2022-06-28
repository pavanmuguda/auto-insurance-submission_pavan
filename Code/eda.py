"""
Created on Thu Jun 23 02:51:09 2022
This script helps to perform EDA on the datasets
@author: pavan
"""


from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import cprint

class EDA:
    """
    Class of two functions for performing EDA
    """
    def __init__(self, df):
        """
        Initialize the EDA class
        :param df: Input Dataset
        """
        self.df = df

    def pca_plot(self):
        """
        This function performs PCA and plots the variance plot
        :return: None
        """
        pca = PCA().fit(self.df)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
        cprint('More than 90% of the training data can be explained by just 15 features', color='grey')

    def pairwise_correlation_heatmap(self):
        """
        This function computes pairwiese correlation coefficients between feature vectors
        and plots them as a heat map
        :return: None
        """
        cprint("Heatmap of dataset", 'green', attrs=["bold"])
        print("-----" * 10)
        plt.figure(figsize=(17, 17))
        sns.heatmap(self.df.corr(), annot=True, fmt='.2f', vmin=-1, vmax=1)
        plt.xticks(rotation=45)

    def outlier_analysis(self):
        pass

