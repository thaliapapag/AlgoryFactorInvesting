from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import csv_to_dataframe
from sklearn.model_selection import GridSearchCV


def neural_net(x, y):
    clf = MLPClassifier(random_state=1, max_iter=300).fit(x, y)
    y_pred = clf.predict(x)
    accuracy = clf.score(x, y)
    return y_pred, accuracy


def grid_search(xTrain, yTrain):

    clf = MLPClassifier(random_state=1)
    #narrow down these parameters maybe?
    dict = {
        'hidden_layer_sizes': [(30,),(50,)],
        'batch_size':[128,256],
        'activation': ['relu', 'tanh'],
        'alpha':[0.001, 0.01, 0.1],
    }

    grid_search = GridSearchCV(clf, dict, cv=10, n_jobs=-1, verbose=2)

    grid_search.fit(xTrain,yTrain)

    classifier = grid_search.best_estimator_
    bestParams = grid_search.best_params_

    return classifier, bestParams

#correlation matrix between the x columns and the y column
def correlation_matrix(x, y):
    x = x.corr(y)
    return x

def main():
    file = 'SPY_data_2022_2024.csv'
    data = csv_to_dataframe.csv_to_dataframe(file)

    alphas_df = csv_to_dataframe.csv_to_dataframe('alphas_SPY_2022_2024.csv')

    y = csv_to_dataframe.create_y(data, alphas_df, days_ago=7)

    #y = y['returns_x_days_ago'].values.flatten()

    x = alphas_df

    corr = correlation_matrix(x, y)
    print('***** correlation *******')
    print(corr)

    # y_pred, accuracy = neural_net(x, y)
    
    # print('***** y *******')
    # print(y_pred)
    # print('***** accuracy *******')
    # print(accuracy)

    # best, best_params = grid_search(x, y)

    # print("****************")
    # print(best_params)
    # print("BEST**************")
    # print(best)


if __name__ == "__main__":
    main()
