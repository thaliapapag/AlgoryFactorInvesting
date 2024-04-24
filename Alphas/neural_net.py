from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
import pandas as pd
import numpy as np
import csv_to_dataframe
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def perceptron(xTrain, yTrain, xTest, yTest):
    clf = Perceptron().fit(xTrain, yTrain)
    y_pred = clf.predict(xTest)
    accuracy = clf.score(xTest, yTest)
    weights = clf.coef_
    return y_pred, accuracy, weights

def neural_net(xTrain, yTrain, xTest, yTest):
    clf = MLPClassifier(random_state=1, max_iter=300).fit(xTrain, yTrain)
    y_pred = clf.predict(xTest)
    accuracy = clf.score(xTest, yTest)
    return y_pred, accuracy

def decision_tree(xTrain, yTrain, xTest, yTest):
    clf = DecisionTreeRegressor(random_state=1).fit(xTrain, yTrain)
    y_pred = clf.predict(xTest)
    accuracy = clf.score(xTest, yTest)
    return y_pred, accuracy

def linear_regression(xTrain, yTrain, xTest, yTest):
    clf = LinearRegression().fit(xTrain, yTrain)
    y_pred = clf.predict(xTest)
    accuracy = clf.score(xTest, yTest)
    return y_pred, yTest, accuracy



#grid search is currently not working
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

# def correlation_matrix(x, y):
#     x['y'] = y
#     corr = x.corr()
#     x.drop('y', axis=1, inplace=True)
#     return corr


def feature_output_correlation(x, y):
    correlations = {}
    lowest_correlation_column = ''
    lowest_correlation = 1
    highest_correlation_column = ''
    highest_correlation = 0
    for column in x.columns:
        # Calculate the correlation between the feature column and the target variable
        correlation = x[column].corr(y.iloc[:, 0])  # Assuming 'y' has a single target column
        correlations[column] = correlation

        if (correlation < lowest_correlation):
            print("minimum")
            lowest_correlation_column = column
            lowest_correlation = correlation
        if (correlation > highest_correlation):
            highest_correlation_column = column
            highest_correlation = correlation
    
    print(f"The lowest correlation with the output is {lowest_correlation_column} at {lowest_correlation}. The highest correlation with the output is {highest_correlation_column} at {highest_correlation}")
    
    return pd.Series(correlations)

def correlation_matrix(correlations):
    correlations = pd.DataFrame(correlations, columns=['Correlation'])
    correlations.index.name = 'Feature'

    correlations.reset_index(inplace=True)

    print(correlations)

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations.pivot(index='Feature', columns='Feature', values='Correlation'), cmap='coolwarm', annot=True)

    plt.title('Feature-Output Correlation Heatmap')
    plt.show()


# def plot_correlation_matrix(x, y):
#     corr = correlation_matrix(x, y)
#     sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
#     plt.title('Correlation Matrix')
#     plt.show()

def main():
    # # file = 'SPY_data_2022_2024.csv'
    # # data = csv_to_dataframe.csv_to_dataframe(file)

    # alphas_df = csv_to_dataframe.csv_to_dataframe('/Users/peterloiselle/AlgoryFactorInvesting/newalphas_SPY_2022_2024.csv') 

    # y = csv_to_dataframe.csv_to_dataframe('/Users/peterloiselle/AlgoryFactorInvesting/y_SPY_2022_2024.csv')

    # #y = y['returns_x_days_ago'].values.flatten()

    # alpha_output_correlations = feature_output_correlation(alphas_df, y)
    # # print(alpha_output_correlations)

    # # correlation_matrix(alpha_output_correlations)

    # # y_pred, accuracy = neural_net(x, y)
    
    # # print('***** y *******')
    # # print(y_pred)
    # # print('***** accuracy *******')
    # # print(accuracy)

    # # best, best_params = grid_search(x, y)

    # # print("****************")
    # # print(best_params)
    # # print("BEST**************")
    # # print(best)
    alphas_df = pd.read_csv('newalphas_SPY_2022_2024.csv')
    y = pd.read_csv('y_SPY_2022_2024.csv')
    xTrain, xTest, yTrain, yTest = train_test_split(alphas_df, y['Close'], test_size=0.3, random_state=42)
    # #### where the models are done ######
    # # print(linear_regression(xTrain, yTrain, xTest, yTest))
    # # print(neural_net(xTrain, yTrain, xTest, yTest))
    # # print(decision_tree(xTrain, yTrain, xTest, yTest))
    y_pred, accuracy, weights = perceptron(xTrain, yTrain, xTest, yTest)
    print(y_pred)
    print(accuracy)
    print(weights)



if __name__ == "__main__":
    main()
