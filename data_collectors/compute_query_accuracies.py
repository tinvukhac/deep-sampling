import numpy as np
import pandas as pd


def load_query_parameters(dataset):
    params_f = open('../data/query_logs/query_params/params_{}'.format(dataset))
    lines = params_f.readlines()

    ratios = [float(x) for x in lines[0].strip().split(',')]
    budgets = [float(x) for x in lines[1].strip().split(',')]

    ratios_str = ','.join('%.6f' % x for x in ratios)
    budgets_str = ','.join('%.6f' % x for x in budgets)

    ratios = [float(x) for x in ratios_str.strip().split(',')]
    budgets = [float(x) for x in budgets_str.strip().split(',')]

    return ratios, budgets


def extract_acc(dataset, input_filename, output_filename):
    ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    # ratios = [0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
    # budgets = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
    #            0.18, 0.19, 0.2]
    budgets = [0.00001, 0.000015, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.0001,
               0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005,
               0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095]

    # Load query parameters
    # ratios, budgets = load_query_parameters(dataset)

    df = pd.read_csv(input_filename, header=None)
    df = df.iloc[:, 1:]
    data = pd.DataFrame.to_numpy(df)
    print(data.shape)

    estimated_counts = {}
    for i in range(data.shape[0] - 10):
        budget = data[i][0]
        ratio = data[i][1]
        key = '{:.6f}-{:.6f}'.format(budget, ratio)
        estimated_counts[key] = data[i][2:]

    actual_counts = {}
    for i in range(data.shape[0] - 10, data.shape[0]):
        budget = data[i][0]
        ratio = data[i][1]
        key = '{:.6f}-{:.6f}'.format(budget, ratio)
        actual_counts[key] = data[i][2:]

    for budget in budgets:
        for ratio in ratios:
            key = '{:.6f}-{:.6f}'.format(budget, ratio)
            # key = "%1.6f-%1.6f" % budget, ratio
            print (key)
            actual_key = '{:.6f}-{:.6f}'.format(1.0, ratio)
            # full_budget = 1.0
            # actual_key = "%1.6f-%1.6f" % full_budget, ratio
            print (actual_key)
            estimated_count = estimated_counts[key]
            actual_count = actual_counts[actual_key]
            diff = abs(estimated_count / budget - actual_count)
            acc = [1.0 if diff[i] == 0 else max(0.0, 1.0 - diff[i] / actual_count[i]) for i in range(len(diff))]
            # acc = max(0, 1 - abs(estimated_count - actual_count) / actual_count)
            estimated_counts[key] = np.mean(acc)

    output_f = open(output_filename, 'w')
    for budget in budgets:
        for ratio in ratios:
            key = '{:.6f}-{:.6f}'.format(budget, ratio)
            print ('{}, acc = {}'.format(key, estimated_counts[key]))
            output_f.writelines('{},{},{},{}\n'.format(dataset, budget, ratio, estimated_counts[key]))

    output_f.close()


def main():
    print("Computing query accuracies")
    f = open('../data/all_datasets.txt')
    lines = f.readlines()

    for line in lines:
        dataset = line.strip()
        extract_acc(dataset, '../data/query_logs/results_05_27/result_{}'.format(dataset), '../data/query_logs/query_accuracies_05_27/acc_{}'.format(dataset))

    # Concatenate the result to 1 file
    filenames = ['../data/query_logs/query_accuracies_05_27/acc_{}'.format(line.strip()) for line in lines]
    acc_filename = '../data/query_accuracies_05_27.csv'
    acc_f = open(acc_filename, 'w')
    for filename in filenames:
        infile = open(filename)
        for line in infile:
            acc_f.write(line)
        infile.close()
    acc_f.close()

    # filenames = ['lakes_1', 'lakes_2', 'lakes_3', 'lakes_4']
    #
    # for dataset in filenames:
    #     extract_acc(dataset, '../data/real_datasets/lakes/query_logs/results/result_{}'.format(dataset),
    #                 '../data/real_datasets/lakes/query_logs/query_accuracies/acc_{}'.format(dataset))
    #
    # # Concatenate the result to 1 file
    # filenames = ['../data/real_datasets/lakes/query_logs/query_accuracies/acc_{}'.format(dataset) for dataset in filenames]
    # acc_filename = '../data/lakes_query_accuracies.csv'
    # acc_f = open(acc_filename, 'w')
    # for filename in filenames:
    #     infile = open(filename)
    #     for line in infile:
    #         acc_f.write(line)
    #     infile.close()
    # acc_f.close()


if __name__ == "__main__":
    main()
