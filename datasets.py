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
        hist = hist / hist.max()
        hist = hist.reshape((hist.shape[0], hist.shape[1], 1))
        histograms.append(hist)

    return np.array(histograms)
