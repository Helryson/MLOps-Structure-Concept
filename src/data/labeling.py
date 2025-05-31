import pandas as pd
from tensorflow.keras.utils import to_categorical

def map_label(df):
    '''Faz o mapeamento de cada label dentro do dataframe para seu respectivo texto'''

    label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    df['mapped_label'] = df['label'].map(label_map)
    return df

def encode(df, num_classes=4):
    '''Usado em labeling e não em build_features pois são valores de target(y)'''
    return to_categorical(df['label'], num_classes=num_classes)