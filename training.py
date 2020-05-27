import datasets
import models
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import time


def main():
    print ("Train and test a prediction model for SE problem")

    # Load dataset attributes
    num_rows = 16
    num_columns = 16
    query_attributes = datasets.load_query_attributes('data/query_accuracies.csv')
    histogram_dir = 'data/histogram_values/{}x{}'.format(num_rows, num_columns)
    histograms = datasets.load_histograms(query_attributes, histogram_dir)

    train_query_attributes, test_query_attributes, train_histograms, test_histograms = train_test_split(query_attributes, histograms, test_size=0.25, random_state=42)

    # pd.DataFrame.to_csv(train_query_attributes, 'data/train_query_attributes.csv')
    # pd.DataFrame.to_csv(test_query_attributes, 'data/test_query_attributes.csv')

    # train_query_attributes = pd.read_csv('data/train_query_attributes.csv')
    # test_query_attributes = pd.read_csv('data/test_query_attributes.csv')
    # print (test_query_attributes.shape)
    # acc_predictions = pd.read_csv('data/acc_predictions.csv', delimiter=',', header=None, names=['pred'])
    # print (acc_predictions.shape)
    # df = pd.concat([test_query_attributes, acc_predictions], axis=1)
    # print (df.shape)
    # pd.DataFrame.to_csv(df, 'data/acc_pred.csv')
    # return

    train_query_x = pd.DataFrame.to_numpy(train_query_attributes[['sampling_budget', 'query_ratio']])
    test_query_x = pd.DataFrame.to_numpy(test_query_attributes[['sampling_budget', 'query_ratio']])
    train_y = train_query_attributes['mean_accuracy']
    test_y = test_query_attributes['mean_accuracy']

    # train_query_x = pd.DataFrame.to_numpy(train_query_attributes[['query_ratio', 'mean_accuracy']])
    # test_query_x = pd.DataFrame.to_numpy(test_query_attributes[['query_ratio', 'mean_accuracy']])
    # train_y = train_query_attributes['sampling_budget']
    # test_y = test_query_attributes['sampling_budget']

    # Create the MLP and CNN models
    mlp = models.create_mlp(train_query_x.shape[1], regress=False)
    cnn = models.create_cnn(num_rows, num_columns, 1, regress=False)

    # Create the input to our final set of layers as the *output* of both the MLP and CNN
    combined_input = concatenate([mlp.output, cnn.output])

    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(4, activation="relu")(combined_input)
    x = Dense(1, activation="linear")(x)

    # our final model will accept categorical/numerical data on the MLP
    # input and images on the CNN input, outputting a single value (the
    # predicted budget)
    model = Model(inputs=[mlp.input, cnn.input], outputs=x)

    # compile the model using mean absolute percentage error as our loss,
    # implying that we seek to minimize the absolute percentage difference
    # between our budget *predictions* and the *actual budgets*
    EPOCHS = 20
    LR = 1e-2
    # opt = Adam(lr=1e-4, decay=1e-4 / 200)
    opt = Adam(lr=LR, decay=LR/EPOCHS)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    # train the model
    print("[INFO] training model...")
    start_time = time.time()
    model.fit(
        [train_query_x, train_histograms], train_y,
        validation_data=([test_query_x, test_histograms], test_y),
        epochs=EPOCHS, batch_size=128)
    end_time = time.time()
    print("--- Trained in %s seconds ---" % (end_time - start_time))

    # make predictions on the testing data
    print("[INFO] predicting sampling budget ratio...")
    preds = model.predict([test_query_x, test_histograms])

    np.savetxt('data/budget_predictions.csv', preds)

    # compute the difference between the *predicted* sampling budget ratio and the
    # *actual* budget, then compute the percentage difference and
    # the absolute percentage difference
    diff = preds.flatten() - test_y
    percent_diff = (diff / test_y) * 100
    abs_percent_diff = np.abs(percent_diff)

    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    print ("Synthetic dataset:")
    print ('mean = {}, std = {}'.format(mean, std))

    # Test on real dataset
    real_query_attributes = datasets.load_query_attributes('data/lakes_query_accuracies.csv')
    real_histogram_dir = 'data/real_datasets/lakes/histogram_values/{}x{}'.format(num_rows, num_columns)
    test_histograms = datasets.load_histograms(real_query_attributes, real_histogram_dir)

    test_query_x = pd.DataFrame.to_numpy(real_query_attributes[['sampling_budget', 'query_ratio']])
    test_y = real_query_attributes['mean_accuracy']

    # test_query_x = pd.DataFrame.to_numpy(real_query_attributes[['query_ratio', 'mean_accuracy']])
    # test_y = real_query_attributes['sampling_budget']

    preds = model.predict([test_query_x, test_histograms])

    # compute the difference between the *predicted* sampling budget ratio and the
    # *actual* budget, then compute the percentage difference and
    # the absolute percentage difference
    diff = preds.flatten() - test_y
    percent_diff = (diff / test_y) * 100
    abs_percent_diff = np.abs(percent_diff)

    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    print ("Real dataset:")
    print ('mean = {}, std = {}'.format(mean, std))


if __name__ == "__main__":
    main()
