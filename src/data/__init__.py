# src/data/__init__.py

from .ingestion import save_load_data
from .cleaning import limpar_texto
from .labeling import map_label, encode
from .splitting import split_text
from .build_features import extract_features

# Agora, ao importar o pacote src.data, você pode acessar essas funções diretamente:
# ex: from src.data import limpar_texto