import pandas as pd
import numpy as np

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def trainingExamples(path_to_csv, training_examples, label, intercept_name):
    df = pd.read_csv(path_to_csv)
    m = len(df.index)
    df2 = pd.DataFrame(np.ones(m), columns=(intercept_name,))
    df.insert(1, intercept_name, df2[intercept_name])
    features = list(df.columns)
    features.remove(label)
    rows = df.index
    np.random.shuffle(list(rows))
    df.reindex(rows)
    X, y = df[:training_examples].ix[:, features[0]:features[-1]], df[:training_examples][label]
    Xval, yval = df[training_examples:].ix[:, features[0]:features[-1]], df[training_examples:][label]
    return df, X, y, Xval, yval

