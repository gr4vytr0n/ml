from numpy import var


def variances(data):
    return var(data, axis=0).argsort()
