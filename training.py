import datasets


def main():
    print ("Train and test a prediction model for SE problem")
    df = datasets.load_query_attributes('data/query_accuracies.csv')
    print (df)


if __name__ == "__main__":
    main()
