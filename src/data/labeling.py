from tensorflow.keras.utils import to_categorical  # type: ignore

def map_label(df):
    """Mapeia labels numéricos para categorias textuais no DataFrame."""
    label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    df['mapped_label'] = df['label'].map(label_map)
    return df

def encode(df, num_classes=4):
    """Converte labels numéricos em one-hot encoding para uso como target."""
    return to_categorical(df['label'], num_classes=num_classes)