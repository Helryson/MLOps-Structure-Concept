import pandas as pd

def map_label(df):
    label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    df['mapped_label'] = df['label'].map(label_map)
    return df