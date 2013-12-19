import pandas as pd
import numpy as np

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def trainingExamples(path_to_csv, training_examples, intercept_name):
    df = pd.read_csv(path_to_csv)
    m = len(df.index)
    df2 = pd.DataFrame(np.ones(m), columns=(intercept_name,))
    df.insert(1, intercept_name, df2[intercept_name])
    rows = df1.index
    np.random.shuffle(list(rows))
    df.reindex(rows)
    return df, df[:training_examples], df[training_examples:]

