import datasets


def main():
    print ("Train and test a prediction model for SE problem")
    df = datasets.load_query_attributes('data/query_accuracies.csv')
    histograms = datasets.load_histograms(df, 16, 16)


if __name__ == "__main__":
    main()
