import numpy as np
import pandas as pd


def load_query_attributes(filename):
    cols = ['dataset_name', 'sampling_budget', 'query_ratio', 'mean_accuracy']
    df = pd.read_csv(filename, delimiter=',', header=None, names=cols)
    # df = df.drop(['dataset_name'], axis=1)
    return df


def load_histograms(df, num_rows, num_columns):
    histograms = []

    histogram_dir = 'data/histogram_values/{}x{}'.format(num_rows, num_columns)

    for filename in df['dataset_name']:
        hist = np.genfromtxt('{}/{}'.format(histogram_dir, filename), delimiter=',')
        histograms.append(hist)

    return histograms
