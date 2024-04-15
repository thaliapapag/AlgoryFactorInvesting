from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import csv_to_dataframe

#have to do neural_net instead of perceptron because perceptron is only for binary classification but is that best?
def neural_net(x, y):
    # x = x.to_frame()
    # y = y.to_frame()
    clf = MLPClassifier(random_state=1, max_iter=300).fit(x, y)  #TODO: change random state after testing is over possibly? 
    return clf


def main():
    

if __name__ == "__main__":
    main()
