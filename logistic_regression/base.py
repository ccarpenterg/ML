import pandas as pd
import numpy as np

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def trainingExamples(path_to_csv, training_examples):
    df = pd.read_csv(path_to_csv)
    m = len(df.index)
    df2 = pd.DataFrame(np.ones(m), columns=('pixel',))
    df.insert(1, 'pixel', df2['pixel'])
    rows = df1.index
    np.random.shuffle(list(rows))
    df.reindex(rows)
    return df, df[:training_examples], df[training_examples:]

