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
    budgets = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
               0.18, 0.19, 0.2]

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
        key = '{}-{}'.format(budget, ratio)
        estimated_counts[key] = data[i][2:]

    actual_counts = {}
    for i in range(200, 210):
        budget = data[i][0]
        ratio = data[i][1]
        key = '{}-{}'.format(budget, ratio)
        actual_counts[key] = data[i][2:]

    for budget in budgets:
        for ratio in ratios:
            key = '{}-{}'.format(budget, ratio)
            actual_key = '{}-{}'.format(1.0, ratio)
            estimated_count = estimated_counts[key]
            actual_count = actual_counts[actual_key]
            diff = abs(estimated_count / budget - actual_count)
            acc = [1.0 if diff[i] == 0 else max(0.0, 1.0 - diff[i] / actual_count[i]) for i in range(len(diff))]
            # acc = max(0, 1 - abs(estimated_count - actual_count) / actual_count)
            estimated_counts[key] = np.mean(acc)

    output_f = open(output_filename, 'w')
    for budget in budgets:
        for ratio in ratios:
            key = '{}-{}'.format(budget, ratio)
            print ('{}, acc = {}'.format(key, estimated_counts[key]))
            output_f.writelines('{},{},{},{}\n'.format(dataset, budget, ratio, estimated_counts[key]))

    output_f.close()


def main():
    print("Computing query accuracies")
    f = open('../data/all_datasets.txt')
    lines = f.readlines()

    for line in lines:
        dataset = line.strip()
        extract_acc(dataset, '../data/query_logs/results_05_09/result_{}'.format(dataset), '../data/query_logs/query_accuracies/acc_{}'.format(dataset))

    # Concatenate the result to 1 file
    filenames = ['../data/query_logs/query_accuracies/acc_{}'.format(line.strip()) for line in lines]
    acc_filename = '../data/query_accuracies.csv'
    acc_f = open(acc_filename, 'w')
    for filename in filenames:
        infile = open(filename)
        for line in infile:
            acc_f.write(line)
        infile.close()
    acc_f.close()


if __name__ == "__main__":
    main()
