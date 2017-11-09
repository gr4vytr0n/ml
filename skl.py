from numpy import array
from sklearn.neighbors import KNeighborsClassifier

dataset = {
    'data': array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]),
    'target': array([0, 0, 1, 1]),
    'target_names': array(['A', 'B'])
}

data_X = dataset['data']
data_y = dataset['target']
data_labels = dataset['target_names']

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(data_X,data_y)

predict = neigh.predict([[0,0.5]])

predict_label = data_labels[int(predict[0])]
print(predict_label)
