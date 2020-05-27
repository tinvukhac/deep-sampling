
def main():
    print('Group query accuracy data by distribution')
    distributions = ['combo', 'diagonal', 'gauss', 'parcel', 'uniform']

    combo_f = open('../data/query_accuracies_combo.csv', 'w')
    diagonal_f = open('../data/query_accuracies_diagonal.csv', 'w')
    gauss_f = open('../data/query_accuracies_gauss.csv', 'w')
    parcel_f = open('../data/query_accuracies_parcel.csv', 'w')
    uniform_f = open('../data/query_accuracies_uniform.csv', 'w')

    input_f = open('../data/query_accuracies_all.csv')
    line = input_f.readline()
    while line:
        if 'Combo' in line:
            combo_f.writelines(line)
        elif 'Diagonal' in line:
            diagonal_f.writelines(line)
        elif 'Gauss' in line:
            gauss_f.writelines(line)
        elif 'Parcel' in line:
            parcel_f.writelines(line)
        elif 'Uniform' in line:
            uniform_f.writelines(line)

        line = input_f.readline()

    combo_f.close()
    diagonal_f.close()
    gauss_f.close()
    parcel_f.close()
    uniform_f.close()
    input_f.close()


if __name__ == '__main__':
    main()
