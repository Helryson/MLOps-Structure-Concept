import logging
import click
from data.database.connection import PostgresDB

from src.data import (
    save_load_data,
    limpar_texto,
    map_label,
    encode,
    split_text,
    extract_features
)

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

@click.command()
@click.argument('output_raw_path', type=click.Path(exists=True))
def main(output_raw_path):
    """Executa pipeline de processamento dos dados."""

    # Inicializa objeto para conexão com banco PostgreSQL
    db = PostgresDB()

    # Estabelece conexão e cria tabela para armazenar dados processados
    conn = db.conectar()
    db.criar_tabela()

    # Configura logger para registrar informações durante a execução
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # Carrega dataset bruto do caminho informado, salvando versão raw no disco
    logger.info("Carregando os dados...")
    df = save_load_data(output_raw_path, logger)
    logger.info(f'Raw dataset salvo com sucesso em {output_raw_path}')

    # Aplica limpeza e tokenização no texto do dataset
    logger.info("Limpando e tokenizando textos...")
    df = limpar_texto(df)

    # Mapeia os rótulos textuais para valores numéricos
    logger.info("Mapeando os rótulos...")
    df = map_label(df)

    # Insere os dados processados no banco de dados
    db.inserir_dados(df, conn)

    # Separa features (texto limpo) e labels codificados
    X = df['cleaned_text']
    y = encode(df)

    # Divide os dados em treino e teste, usando 30% para teste
    logger.info("Dividindo os dados em treino e teste...")
    X_train, X_test = split_text(X, y, test_size=0.3)

    # Extrai features dos dados textuais para uso em modelos
    logger.info("Extraindo features do texto...")
    extract_features(X_train, X_test)

    logger.info("Pipeline executada com sucesso!")

if __name__ == '__main__':
    main()
