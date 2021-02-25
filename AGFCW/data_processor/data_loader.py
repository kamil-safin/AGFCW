import pandas as pd
import numpy as np

class MSDDataset:

    def __init__(
        self,
        path='data/YearPredictionMSD.txt',
        normalize=True,
        binarize=True,
        train_size=463715,
        shuffle=True
    ) -> None:
        self.path = path
        self.normalize = normalize
        self.binarize = binarize
        self.train_size = train_size
        self.shuffle = shuffle

    def load(self):
        data = pd.read_csv(self.path, header=None, usecols=range(13))
        y = data[0].to_numpy()
        if self.binarize:
            y = (y < 2001).astype(int)
        data.drop(0, axis=1, inplace=True)
        X = data.to_numpy()
        self.X_train = X[:self.train_size]
        self.y_train = y[:self.train_size]
        self.X_test = X[self.train_size:]
        self.y_test = y[self.train_size:]
        if self.normalize:
            self.X_train = self._normalize(self.X_train)
            self.X_test = self._normalize(self.X_test)
        self.train_idxs = list(range(self.X_train.shape[0]))
        if self.shuffle:
            np.random.shuffle(self.train_idxs)
        self.dim = X.shape[1]

    def get_train_data(self, idxs):
        return np.take(self.X_train, idxs, axis=0), np.take(self.y_train, idxs)

    def get_test_data(self):
        return self.X_test, self.y_test

    def _normalize(self, x):
        mean_x = np.mean(x, axis=0)
        std_x = np.std(x, axis=0)
        return (x - mean_x) / std_x