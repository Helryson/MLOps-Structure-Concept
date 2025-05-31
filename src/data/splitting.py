from sklearn.model_selection import train_test_split

def split_text(X, y, test_size = 0.2):

    '''Separa o dataframe '''
    return train_test_split(X, y, test_size=test_size, random_state=42)