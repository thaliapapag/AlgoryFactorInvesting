from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

def alphas_to_df()

def neural_net(x, y):
    # x = x.to_frame()
    # y = y.to_frame()
    clf = MLPClassifier(random_state=1, max_iter=300).fit(x, y)  #TODO: change random state after testing is over possibly? 
    return clf

def main():
    x = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)))
    y = pd.DataFrame(np.random.randint(0, 2, size=(100, 1)))
    print(neural_net(x, y))

if __name__ == "__main__":
    main()
