import datasets

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def run_linear_regression_model(train_query_x, test_query_x, train_y, test_y, real_x, real_y):
    reg_map = LinearRegression().fit(train_query_x, train_y)
    pred_y = reg_map.predict(test_query_x)

    diff = pred_y.flatten() - test_y
    percent_diff = (diff / test_y) * 100
    abs_percent_diff = np.abs(percent_diff)
    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    print ('mean = {}, std = {}'.format(mean, std))

    pred_y = reg_map.predict(real_x)

    diff = pred_y.flatten() - real_y
    percent_diff = (diff / test_y) * 100
    abs_percent_diff = np.abs(percent_diff)
    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)
    print ('mean = {}, std = {}'.format(mean, std))


def main():
    print ("Train and test a prediction model for SE problem using baseline model")

    query_attributes = datasets.load_query_attributes('data/query_accuracies.csv')

    train_query_attributes, test_query_attributes = train_test_split(
        query_attributes, test_size=0.25, random_state=42)

    # train_query_x = pd.DataFrame.to_numpy(train_query_attributes[['sampling_budget', 'query_ratio']])
    # test_query_x = pd.DataFrame.to_numpy(test_query_attributes[['sampling_budget', 'query_ratio']])
    # train_y = train_query_attributes['mean_accuracy']
    # test_y = test_query_attributes['mean_accuracy']

    train_query_x = pd.DataFrame.to_numpy(train_query_attributes[['query_ratio', 'mean_accuracy']])
    test_query_x = pd.DataFrame.to_numpy(test_query_attributes[['query_ratio', 'mean_accuracy']])
    train_y = train_query_attributes['sampling_budget']
    test_y = test_query_attributes['sampling_budget']

    real_query_attributes = datasets.load_query_attributes('data/lakes_query_accuracies.csv')

    # real_x = pd.DataFrame.to_numpy(real_query_attributes[['sampling_budget', 'query_ratio']])
    # real_y = real_query_attributes['mean_accuracy']

    real_x = pd.DataFrame.to_numpy(real_query_attributes[['query_ratio', 'mean_accuracy']])
    real_y = real_query_attributes['sampling_budget']

    run_linear_regression_model(train_query_x, test_query_x, train_y, test_y, real_x, real_y)


if __name__ == '__main__':
    main()