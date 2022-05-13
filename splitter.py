import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def splitter(df, target, train_split_1 = .8, train_split_2 = .7, random_state = 123):
    
    train, test = train_test_split(df, train_size = train_split_1, random_state = random_state, stratify = df[target])
    
    train, validate = train_test_split(train, train_size = train_split_2, random_state = random_state, stratify = df[target])

    return train, validate, test