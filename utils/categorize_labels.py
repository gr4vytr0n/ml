from sklearn import preprocessing

# build array of number representing how labels
# are applied to dataset


def categorize_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)

    # returns array of numbers (as labels indices)
    # and an array providing a labels
    return le.transform(labels), le.classes_
