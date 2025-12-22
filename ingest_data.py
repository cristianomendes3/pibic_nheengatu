import pandas as pd
from datasets import Dataset
import os
import sys

# Configuração dos caminhos (Paths)
ARQUIVO_ENTRADA = "dados_iniciais_nheengatu.xlsx"
ARQUIVO_SAIDA = "dataset_nheengatu_raw"

def ingest_data():
  """
  Lê o arquivo Excel, valida as colunas e converte para o formato Hugging Face Dataset.
  """
  print(f"[INFO] Iniciando ingestão do arquivo: {ARQUIVO_ENTRADA}")

  #1. Verificação de existência do arquivo
  if not os.path.exists(ARQUIVO_ENTRADA):
    print(f"[ERRO] O arquivo {ARQUIVO_ENTRADA} não foi encontrado.")
    print("Dica: Crie um Excel com colunas 'Palavra' e 'Significado' para teste.")
    sys.exit(1)

  #2. Carregamento com Pandas (engine='openpyxl' é necessário para .xlsx)
  try:
    df = pd.read_excel(ARQUIVO_ENTRADA, engine='openpyxl')
  except Exception as e:
    print(f"[ERRO] Ocorreu um erro ao carregar o arquivo {ARQUIVO_ENTRADA}: {e}")
    sys.exit(1)

  #3. Verificação de Colunas
  if 'Palavra' not in df.columns or 'Significado' not in df.columns:
    print(f"[ERRO] As colunas 'Palavra' e 'Significado' são obrigatórias no arquivo {ARQUIVO_ENTRADA}.")
    print(f"Colunas encontradas: {df.columns}")
    print("Ajuste o cabeçalho do Excel e tente novamente.")
    sys.exit(1)

  print(f"[INFO] Colunas validadas. Total de registros: {len(df)}")

  # Exibir uma amostra para garantir que não há caracteres estranhos (encoding)
  print("\n--- Amostra dos Dados ---")
  print(df.head())
  print("-------------------------\n")

  #4. Conversão para o formato Hugging Face Dataset
  print("[INFO] Convertendo para o formato Hugging Face Dataset...")
  try:
    hf_dataset = Dataset.from_pandas(df)

    # Opcional: Salvar em disco no formato nativo do Arrow para carregamento rápido depois
    hf_dataset.save_to_disk(ARQUIVO_SAIDA)

    print(f"[INFO] Dataset convertido e salvo na pasta '{ARQUIVO_SAIDA}'.")
    print(hf_dataset)
  except Exception as e:
    print(f"[ERRO] Ocorreu um erro ao converter o DataFrame para o Dataset: {e}")
    sys.exit(1)

if __name__ == "__main__":
  ingest_data()