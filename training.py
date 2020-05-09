import datasets
import models
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import argparse
import locale
import os


def main():
    print ("Train and test a prediction model for SE problem")

    # Load dataset attributes
    num_rows = 16
    num_columns = 16
    query_attributes = datasets.load_query_attributes('data/query_accuracies.csv')
    histograms = datasets.load_histograms(query_attributes, num_rows, num_columns)
    # print (histograms)

    train_query_attributes, test_query_attributes, train_histograms, test_histograms = train_test_split(query_attributes, histograms, test_size=0.25, random_state=42)
    train_query_x = pd.DataFrame.to_numpy(train_query_attributes[['sampling_budget', 'query_ratio']])
    test_query_x = pd.DataFrame.to_numpy(test_query_attributes[['sampling_budget', 'query_ratio']])
    train_y = train_query_attributes['mean_accuracy']
    test_y = test_query_attributes['mean_accuracy']

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
    # predicted price of the house)
    model = Model(inputs=[mlp.input, cnn.input], outputs=x)

    # compile the model using mean absolute percentage error as our loss,
    # implying that we seek to minimize the absolute percentage difference
    # between our price *predictions* and the *actual prices*
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    # train the model
    print("[INFO] training model...")
    model.fit(
        [train_query_x, train_histograms], train_y,
        validation_data=([test_query_x, test_histograms], test_y),
        epochs=20, batch_size=8)

    # make predictions on the testing data
    print("[INFO] predicting house prices...")
    preds = model.predict([test_query_x, test_histograms])

    # compute the difference between the *predicted* house prices and the
    # *actual* house prices, then compute the percentage difference and
    # the absolute percentage difference
    diff = preds.flatten() - test_y
    percent_diff = (diff / test_y) * 100
    abs_percent_diff = np.abs(percent_diff)

    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)
    print ('mean = {}, std = {}'.format(mean, std))


if __name__ == "__main__":
    main()
