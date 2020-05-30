import os
import numpy as np


def execute_queries():
    print("Computing dataset histogram")
    f = open('../data/all_datasets.txt')
    lines = f.readlines()

    histogram_sizes = ['16,16', '32,32']
    histogram_dirs = ['16x16', '32x32']
    for i in range(len(histogram_sizes)):
        for line in lines:
            filename = line.strip()
            print (
                'spark-submit --master spark://ec-hn.cs.ucr.edu:7077 beast-uber-spark-0.0.1-SNAPSHOT.jar histogram deepspatial_2020/datasets/{} deepspatial_2020/histograms/{}/{} \'iformat:wkt(0)\' shape:{} -overwrite'.format(
                    filename, histogram_dirs[i], filename, histogram_sizes[i]))

    f.close()


def extract_histogram(input_filename, output_filename, num_rows, num_columns):
    hist = np.zeros((num_rows, num_columns))
    input_f = open(input_filename)

    line = input_f.readline()
    line = input_f.readline()
    while line:
        data = line.strip().split('\t')
        column = int(data[0])
        row = int(data[1])
        freq = int(data[3])
        hist[row][column] = freq

        line = input_f.readline()

    np.savetxt(output_filename, hist.astype(int), fmt='%i', delimiter=',')

    input_f.close()


def extract_histograms():
    # histogram_dirs = ['16x16', '32x32', '64x64']
    # f = open('../data/all_datasets.txt')
    # lines = f.readlines()
    # filenames = [line.strip() for line in lines]

    histogram_dirs = ['64x64']
    filenames = ['lakes_1', 'lakes_2', 'lakes_3', 'lakes_4']

    for histogram_dir in histogram_dirs:
        data = histogram_dir.split('x')
        num_rows = int(data[0])
        num_columns = int(data[1])
        input_dir = '../data/real_datasets/lakes/histograms/{}'.format(histogram_dir)
        output_dir = '../data/real_datasets/lakes/histogram_values/{}'.format(histogram_dir)
        for filename in filenames:
            extract_histogram('{}/{}'.format(input_dir, filename), '{}/{}'.format(output_dir, filename), num_rows, num_columns)


def shrink_histograms(num_rows, num_columns):
    input_dir = '../data/histogram_values/{}x{}'.format(num_rows, num_columns)
    output_dir = '../data/histogram_values/{}x{}'.format(num_rows / 2, num_columns / 2)

    f = open('../data/all_datasets.txt')
    lines = f.readlines()
    filenames = [line.strip() for line in lines]

    # filenames = ['lakes_1', 'lakes_2', 'lakes_3', 'lakes_4']
    for dataset in filenames:
        hist_input_filename = '{}/{}'.format(input_dir, dataset)
        hist_output_filename = '{}/{}'.format(output_dir, dataset)

        hist_input = np.genfromtxt(hist_input_filename, delimiter=',')
        hist_output = np.zeros((num_rows / 2, num_columns / 2))

        for i in range(hist_output.shape[0]):
            for j in range(hist_output.shape[1]):
                hist_output[i, j] = hist_input[2*i, 2*j] + hist_input[2*i + 1, 2*j] + hist_input[2*i, 2*j + 1] + hist_input[2*i + 1, 2*j + 1]

        np.savetxt(hist_output_filename, hist_output.astype(int), fmt='%i', delimiter=',')


def main():
    # execute_queries()
    # extract_histograms()
    shrink_histograms(2, 2)


if __name__ == "__main__":
    main()
