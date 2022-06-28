"""
Created on Thu Jun 23 02:51:09 2022
This script helps to classify the auto insurance dataset with five
different machine learning models
@author: pavan
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from Code.preprocess import CleanData
from sklearn.metrics import plot_confusion_matrix
from yellowbrick.classifier import ClassPredictionError
from termcolor import cprint
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

class ModelsforAutoInsurance:
    """
    Class of functions used for binary classification of auto insurance claims
    """
    def __init__(self, X, y):
        """
        Initialize the class with train dataset
        :param X: train dataset features
        :param y: train dataset labels
        """
        self.X = X
        self.y = y

    def logistic_regression_model(self):
        """
        This function is used for building logistic regression model pipeline
        :return: pipeline and logistic regression model
        """
        model = LogisticRegression()
        pipeline_logistic = Pipeline(
            [('preprocessing', CleanData()), ('pca', PCA(n_components=15)), ('linear_model', model)])
        pipeline_logistic.fit(self.X, self.y)
        return pipeline_logistic, model

    def decision_tree_classifier(self):
        """
        This function is used for building decision_tree_classifier model pipeline
        :return: pipeline and decision_tree_classifier model
        """
        model = DecisionTreeClassifier()
        pipeline_dtc = Pipeline(
            [('preprocessing', CleanData()), ('pca', PCA(n_components=15)), ('decision_tree', model)])
        pipeline_dtc.fit(self.X, self.y)
        return pipeline_dtc, model

    def svc(self):
        """
        This function is used for building svc model pipeline
        :return: pipeline and svc model
        """
        model = SVC(class_weight='balanced')
        pipeline_svc = Pipeline(
            [('preprocessing', CleanData()), ('pca', PCA(n_components=15)),
             ('svc', model)])
        pipeline_svc.fit(self.X, self.y)
        return pipeline_svc, model

    def random_forest_classifier(self):
        """
        This function is used for building random_forest_classifier model pipeline
        :return: pipeline and random_forest_classifier model
        """
        model = RandomForestClassifier(class_weight = "balanced", random_state = 42)
        pipeline_rand_forst = Pipeline(
            [('preprocessing', CleanData()), ('pca', PCA(n_components=15)),
             ('random forest', model)])
        pipeline_rand_forst.fit(self.X, self.y)
        return pipeline_rand_forst, model

    def knearest_neighbour_classifier(self):
        """
        This function is used for building knearest_neighbour_classifier model pipeline
        :return: pipeline and knearest_neighbour_classifier
        """
        model = KNeighborsClassifier()
        pipeline_knearest = Pipeline(
            [('preprocessing', CleanData()), ('pca', PCA(n_components=15)),
             ('nearest neighbour', model)])
        pipeline_knearest .fit(self.X, self.y)
        return pipeline_knearest, model

    def train_val(self, y_train, y_train_pred, y_test, y_pred):
        """
        This function computes required metrics for the classifier

        :param y_train: reference labels of training dataset
        :param y_train_pred: predicted labels for training dataset by model
        :param y_test: reference labels of testing dataset
        :param y_pred: predicted labels for testing dataset by model
        :return: dataframe with the metrics
        """
        scores = {"train_set": {"Accuracy": accuracy_score(y_train, y_train_pred),
                                "Precision": precision_score(y_train, y_train_pred),
                                "Recall": recall_score(y_train, y_train_pred),
                                "f1": f1_score(y_train, y_train_pred)},

                  "test_set": {"Accuracy": accuracy_score(y_test, y_pred),
                               "Precision": precision_score(y_test, y_pred),
                               "Recall": recall_score(y_test, y_pred),
                               "f1": f1_score(y_test, y_pred)}}

        return pd.DataFrame(scores)

    @classmethod
    def clean_data(cls, dataframe):
        clean_data = CleanData().transform(dataframe)
        perform_pca = PCA(n_components=15).fit_transform(clean_data)
        return perform_pca

    def eval(self, pipeline, X_test, y_test, model):
        """
        This function evaluates chosen classifier using the pipeline
        :param pipeline: pipeline of the model
        :param X_test: test data features
        :param y_test: test data labels
        :param model: model for classifier
        :return: metrics
        """
        y_pred_train = pipeline.predict(self.X)
        y_pred_test = pipeline.predict(X_test)
        cprint('Mertrics and Confusion Matrix of Model', color='green')
        plot_confusion_matrix(pipeline, X_test, y_test, cmap="plasma")
        score_table = self.train_val(self.y, y_pred_train, y_test, y_pred_test)
        # self.class_error_predictor(model, self.clean_data(self.X), self.y, self.clean_data(X_test), y_test)
        return score_table