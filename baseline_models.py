import datasets
import training

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from itertools import combinations


def run_linear_regression_model(train_query_x, test_query_x, train_y, test_y, real_x, real_y):
    start_time = time.time()

    reg_map = LinearRegression().fit(train_query_x, train_y)

    end_time = time.time()
    duration = end_time - start_time
    print("--- Trained in %s seconds ---" % duration)

    pred_y = reg_map.predict(test_query_x)

    diff = pred_y.flatten() - test_y
    percent_diff = (diff / test_y)
    abs_percent_diff = np.abs(percent_diff)

    synthetic_mean = np.mean(abs_percent_diff)
    synthetic_std = np.std(abs_percent_diff)

    print ('mean = {}, std = {}'.format(synthetic_mean, synthetic_std))

    pred_y = reg_map.predict(real_x)

    diff = pred_y.flatten() - real_y
    percent_diff = (diff / test_y)
    abs_percent_diff = np.abs(percent_diff)
    real_mean = np.mean(abs_percent_diff)
    real_std = np.std(abs_percent_diff)
    print ('mean = {}, std = {}'.format(real_mean, real_std))
    return duration, synthetic_mean, synthetic_std, real_mean, real_std


def main():
    print ("Train and test a prediction model for SE problem using baseline model")

    output_f = open('data/output_accuracy_prediction_baseline.csv', 'w')

    query_attributes_files_list = []
    distributions = ['uniform', 'diagonal', 'gauss', 'parcel', 'combo']
    for r in range(1, len(distributions) + 1):
        groups = combinations(distributions, r)
        for g in groups:
            query_attributes_files = []
            for dist in g:
                query_attributes_files.append('data/query_accuracies_{}.csv'.format(dist))
            query_attributes_files_list.append(query_attributes_files)

    query_attributes_all = datasets.load_query_attributes_from_multiple_files(['data/query_accuracies_all.csv'])
    train_query_attributes_all, test_query_attributes_all = train_test_split(
        query_attributes_all, test_size=0.25, random_state=42)

    for query_attributes_files in query_attributes_files_list:
        query_attributes = datasets.load_query_attributes_from_multiple_files(query_attributes_files)

        train_query_attributes, test_query_attributes = train_test_split(
            query_attributes, test_size=0.25, random_state=42)
        real_query_attributes = datasets.load_query_attributes('data/lakes_query_accuracies.csv')

        # train_query_x = pd.DataFrame.to_numpy(train_query_attributes[['sampling_budget', 'query_ratio']])
        # test_query_x = pd.DataFrame.to_numpy(test_query_attributes_all[['sampling_budget', 'query_ratio']])
        # train_y = train_query_attributes['mean_accuracy']
        # test_y = test_query_attributes_all['mean_accuracy']
        # real_x = pd.DataFrame.to_numpy(real_query_attributes[['sampling_budget', 'query_ratio']])
        # real_y = real_query_attributes['mean_accuracy']

        train_query_x = pd.DataFrame.to_numpy(train_query_attributes[['query_ratio', 'mean_accuracy']])
        test_query_x = pd.DataFrame.to_numpy(test_query_attributes_all[['query_ratio', 'mean_accuracy']])
        train_y = train_query_attributes['sampling_budget']
        test_y = test_query_attributes_all['sampling_budget']
        real_x = pd.DataFrame.to_numpy(real_query_attributes[['query_ratio', 'mean_accuracy']])
        real_y = real_query_attributes['sampling_budget']

        duration, synthetic_mean, synthetic_std, real_mean, real_std = run_linear_regression_model(train_query_x, test_query_x, train_y, test_y, real_x, real_y)
        output_f.writelines(
            '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(','.join(query_attributes_files), len(query_attributes_files), duration, synthetic_mean, synthetic_std,
                                              real_mean, real_std))

    output_f.close()

    training.compute_average_accuracy_by_distributions('data/output_accuracy_prediction_baseline.csv', 'data/average_accuracy_by_distributions_baseline_problem2.csv')


if __name__ == '__main__':
    main()
