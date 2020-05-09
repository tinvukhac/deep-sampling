import pandas as pd


def load_query_attributes(filename):
    cols = ['dataset_name', 'sampling_budget', 'query_ratio', 'mean_accuracy']
    df = pd.read_csv(filename, delimiter=',', header=None, names=cols)
    df = df.drop(['dataset_name'], axis=1)
    return df