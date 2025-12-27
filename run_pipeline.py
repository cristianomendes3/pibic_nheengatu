import pandas as pd
import json
from transformers import AutoTokenizer
from normalizer import clean_text_nheengatu
import unicodedata

# Configurações
ARQUIVO_ENTRADA = "100palavras_nheengatu.xlsx"
ARQUIVO_SAIDA_JSON = "relatorio_tokens_nheengatu.json"
MODELO_NOME = "dominguesm/canarim-bert-nheengatu"

def carregar_dados():
  """Carrega a planilha usando pandas (camada de ingestão simplificada)."""
  try:
    # engine='openpyxl' é vital para .xlsx
    df = pd.read_excel(ARQUIVO_ENTRADA, engine='openpyxl')
    # Garante que as colunas existem (normalizando nomes para evitar erro de caixa)
    df.columns = [c.strip().title() for c in df.columns]
    if 'Palavra' not in df.columns or 'Significado' not in df.columns:
      raise ValueError("As colunas 'Palavra' e 'Significado' são obrigatórias.")
    print(f"Planilha '{ARQUIVO_ENTRADA}' carregada com sucesso!")
    return df
  except Exception as e:
    print(f"Erro ao carregar a planilha '{ARQUIVO_ENTRADA}': {e}")
    return None

def processar_pipeline(df, tokenizer):
  """Executa Normalização e Tokenização para cada linha."""
  resultados = []
  stats = {"sucesso": 0, "unk": 0, "total": 0}

  print(f"--- Iniciando processamento de {len(df)} palavras ---")

  for index, row in df.iterrows():
    palavra = str(row['Palavra'])
    significado = str(row['Significado'])

    # 1. Normalização
    # Aplica lowercase, remove pontuação extra, normaliza Unicode (NFC)
    palavra_norm = clean_text_nheengatu(palavra)

    # 2. Tokenização
    tokens = tokenizer.tokenize(palavra_norm)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    # 3. Análise de Qualidade
    # Verifica se o token [UNK] (ID 100 ou similar) apareceu
    tem_unk = tokenizer.unk_token in tokens
    
    if tem_unk:
      stats["unk"] += 1
      status = "ALERTA"
    else:
      stats["sucesso"] += 1
      status = "OK"

    stats["total"] += 1

    # Estrutura para o JSON
    dados_palavra = {
        "original": palavra,
        "processada": palavra_norm,
        "tokens": tokens,
        "ids": ids,
        "significado": significado,
        "status": status
    }

    resultados.append(dados_palavra)
    
    # Log visual rápido no terminal
    print(f"[{status}] {palavra:<15} -> {str(tokens)}")

  return resultados, stats

def main():
  # 1. Carregar Tokenizer
  print ("⏳ Carregando Tokenizer Canarim...")
  try:
    tokenizer = AutoTokenizer.from_pretrained(MODELO_NOME)
    print(f"Tokenizer carregado com sucesso!")
  except Exception as e:
    print(f"Erro ao carregar o Tokenizer: {e}")
    return

  # 2. Carregar Dados
  df = carregar_dados()
  if df is None:
    print(f"Erro ao carregar a planilha '{ARQUIVO_ENTRADA}'. Saindo...")

  # 3. Rodar Pipeline
  resultados, estatisticas = processar_pipeline(df, tokenizer)

  # 4. Gerar Relatório Final
  print("\n" + "="*40)
  print("RELATÓRIO DE PROCESSAMENTO")
  print("="*40)
  print(f"Total de palavras: {estatisticas['total']}")
  print(f"Tokenizadas com sucesso: {estatisticas['sucesso']}")
  print(f"Com tokens desconhecidos [UNK]: {estatisticas['unk']}")
  print(f"Taxa de Sucesso: {(estatisticas['sucesso']/estatisticas['total'])*100:.1f}%")

  # 5. Salvar JSON
  with open(ARQUIVO_SAIDA_JSON, 'w', encoding='utf-8') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
  print(f"\nRelatório salvo em '{ARQUIVO_SAIDA_JSON}'")

if __name__ == "__main__":
  main()