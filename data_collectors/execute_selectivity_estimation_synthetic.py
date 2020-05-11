import os
import random


def main():
    print('Execute selectivity experiment')

    # ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    # budgets = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]

    f = open('../data/all_datasets.txt')
    lines = f.readlines()
    filenames = [line.strip() for line in lines]

    for filename in filenames:
        # Randomize ratios and budget
        num_ratios, max_ratio = 10, 0.1
        ratios = []
        for i in range(num_ratios):
            ratios.append(random.uniform(0, max_ratio))

        num_budgets, max_budget = 20, 0.2
        budgets = []
        for i in range(num_budgets):
            budgets.append(random.uniform(0, max_budget))

        ratios_str = ','.join(str(x) for x in ratios)
        budgets_str = ','.join(str(x) for x in budgets)

        # Save the parameters
        query_params_f = open('query_params/params_{}'.format(filename), 'w')
        query_params_f.writelines('{}\n{}'.format(ratios_str, budgets_str))
        query_params_f.close()

        print(
            'spark-submit --class Driver --master spark://ec-hn.cs.ucr.edu:7077 summarization-project-1.0-SNAPSHOT-jar-with-dependencies.jar deepspatial_2020/datasets/{} {} point selectivity {} > log_{}.txt'.format(
                filename, budgets_str, ratios_str, filename))

    f.close()


if __name__ == "__main__":
    main()
