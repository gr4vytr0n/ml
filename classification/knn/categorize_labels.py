from sklearn import preprocessing

def categorize_labels(labels):
  le = preprocessing.LabelEncoder()
  le.fit(labels)

  return le.transform(labels), le.classes_
