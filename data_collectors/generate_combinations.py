from itertools import combinations


def main():
    print ('Compute all combination of distributions')
    query_attributes_files_list = []
    distributions = ['uniform', 'diagonal', 'gauss', 'parcel', 'combo']
    for r in range(1, len(distributions) + 1):
        groups = combinations(distributions, r)
        for g in groups:
            query_attributes_files = []
            for dist in g:
                query_attributes_files.append('data/query_accuracies_{}.csv'.format(dist))
            query_attributes_files_list.append(query_attributes_files)

    print (query_attributes_files_list)




if __name__ == '__main__':
    main()
