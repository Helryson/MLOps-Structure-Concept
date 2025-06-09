import psycopg2
import os
from dotenv import load_dotenv
import json

load_dotenv()

class PostgresDB:
    def __init__(self):
        # Lê configurações do banco de variáveis de ambiente
        self.config = {
            'host': os.getenv("DB_HOST"),
            'port': os.getenv("DB_PORT"),
            'dbname': os.getenv("DB_NAME"),
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASSWORD")
        }
        self.conn = None

    def conectar(self):
        # Estabelece conexão com o banco PostgreSQL
        try:
            self.conn = psycopg2.connect(**self.config)
            print("Conexão estabelecida com sucesso.")
        except psycopg2.Error as e:
            print("Erro ao conectar ao banco de dados:", e)
            self.conn = None

    def criar_tabela(self):
        # Cria tabela 'noticias_processadas' se não existir
        if self.conn is None:
            print("Conexão não estabelecida.")
            return

        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS noticias_processadas (
                        id SERIAL PRIMARY KEY,
                        text TEXT NOT NULL,
                        label INTEGER NOT NULL,
                        tokens TEXT NOT NULL,
                        tokens_lematizados TEXT NOT NULL,
                        cleaned_text TEXT NOT NULL,
                        mapped_label TEXT NOT NULL
                    );
                """)
                self.conn.commit()
                print("Tabela 'noticias_processadas' criada")
        except psycopg2.Error as e:
            print("Erro ao criar tabela:", e)
            
    def inserir_dados(self, df, conn):
        # Insere registros do DataFrame no banco, validando colunas necessárias
        colunas_esperadas = [
            'text', 'label', 'tokens', 'tokens_lematizados',
            'cleaned_text', 'mapped_label'
        ]

        for coluna in colunas_esperadas:
            if coluna not in df.columns:
                print(f"Coluna ausente no DataFrame: {coluna}")
                return

        try:
            with self.conn.cursor() as cursor:
                for _, linha in df.iterrows():
                    cursor.execute("""
                        INSERT INTO noticias_processadas 
                        (text, label, tokens, tokens_lematizados, cleaned_text, mapped_label)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        linha['text'],
                        int(linha['label']),
                        json.dumps(linha['tokens']),
                        json.dumps(linha['tokens_lematizados']),
                        linha['cleaned_text'],
                        linha['mapped_label']
                    ))
                self.conn.commit()
                print(f"Inseridos {len(df)} registros.")
        except Exception as e:
            print("Erro ao inserir dados:", e)
            self.conn.rollback()
        
        finally:
            # Fecha a conexão após inserção
            self.fechar_conexao()

    def fechar_conexao(self):
        # Encerra a conexão com o banco, se aberta
        if self.conn:
            self.conn.close()
            print("Conexão encerrada.")