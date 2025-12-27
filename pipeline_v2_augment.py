import pandas as pd
import json
import re
from transformers import AutoTokenizer
from normalizer import clean_text_nheengatu

# Configurações de Arquivo
ARQUIVO_ENTRADA_BRUTO = "100palavras_nheengatu_completo.xlsx"
ARQUIVO_SAIDA_JSON = "dataset_nheengatu_expandido.json"
ARQUIVO_SAIDA_CSV = "dataset_nheengatu_expandido.csv" # Útil para inspeção visual no Excel
MODELO_NOME = "dominguesm/canarim-bert-nheengatu"

def carregar_dados_brutos():
  """Carrega a planilha original com suporte a múltiplas abas se necessário."""
  try:
    # engine='openpyxl' é essencial para arquivos .xlsx
    df = pd.read_excel(ARQUIVO_ENTRADA_BRUTO, engine='openpyxl')
    
    # Normalização dos cabeçalhos (remove espaços e converte para Título)
    df.columns = [c.strip().title() for c in df.columns]

    # Validação básica
    if 'Palavra' not in df.columns or 'Significado' not in df.columns:
      raise ValueError("As colunas 'Palavra' e 'Significado' são obrigatórias.")

    print(f"Planilha '{ARQUIVO_ENTRADA_BRUTO}' carregada com sucesso!")
    return df

  except Exception as e:
    print(f"Erro ao carregar a planilha '{ARQUIVO_ENTRADA_BRUTO}': {e}")
    return None

def expandir_linha(row):
  """
  Recebe uma linha do DataFrame e retorna uma lista de dicionários expandidos.
  Realiza o 'Data Augmentation' via produto cartesiano.
  """
  raw_words = str(row['Palavra'])
  raw_meanings = str(row['Significado'])

  # Regex para separar múltiplos itens
  # Separa por vírgula (,), ponto e vírgula (;), barra (/) ou quebra de linha (\n)
  # O \s* remove espaços extras ao redor dos separadores.

  split_patterns = r'[;,/\n]\s*|,\s+'

  lista_palavras = re.split(split_patterns, raw_words)
  lista_significados = re.split(split_patterns, raw_meanings)

  # Limpeza básica (strip) e remoção de itens vazios
  lista_palavras = [w.strip() for w in lista_palavras if w.strip()]
  lista_significados = [m.strip() for m in lista_significados if m.strip()]

  pares_expandidos = []

  # Produto Cartesiano: Cada variante x Cada significado
  for palavra in lista_palavras:
    for significado in lista_significados:
      pares_expandidos.append({
          "palavra_original": palavra, 
          "significado_original": significado,
          "origem_linha": row.name + 2 # +2 para ajustar ao índice do Excel(Header=1, Index=0)
            })

  return pares_expandidos

def processar_augmentacao(df, tokenizer):
  dataset_final = []
  stats = {"original_rows": len(df), "expanded_rows": 0, "unk_tokens": 0}

  print(f"--- Iniciando Augmentação de Dados ---")

  for index, row in df.iterrows():
    # 1. Expansão (Augmentation)
    pares = expandir_linha(row)

    for item in pares:
      # 2. Normalização
      # A palavra é limpa (lowercase, NFC, remove pontuação exceto glotal)
      palavra_norm = clean_text_nheengatu(item['palavra_original'])

      # O significado em português também passa por limpeza leve (opcional)
      significado_clean = item['significado_original'].strip()

      # 3. Tokenização e Validação
      tokens = tokenizer.tokenize(palavra_norm)
      ids = tokenizer.convert_tokens_to_ids(tokens)

      # Verifica se o token [UNK] (ID 100 ou similar) apareceu
      tem_unk = tokenizer.unk_token in tokens
      if tem_unk:
        stats["unk_tokens"] += 1
        status = "ALERTA"
      else:
        status = "OK"

      # Monta o objeto final
      entry = {
          "nheengatu_text": palavra_norm,
          "portuguese_text": significado_clean,
          "tokens": tokens,
          "input_ids": ids,
          "tem_unk": tem_unk,
          "metadata": {
              "raw_nheengatu": item['palavra_original'],
              "source_line": item['origem_linha']
          }
      }
      dataset_final.append(entry)
      stats["expanded_rows"] += 1

      # Log visual rápido no terminal
      print(f"[{status}] {palavra_norm:<15} -> {str(tokens)}")

  return dataset_final, stats

def main():
    # Carregar Tokenizer
    print(f"⏳ Carregando Tokenizer: {MODELO_NOME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODELO_NOME)
    except Exception as e:
        print(f"❌ Erro ao baixar modelo: {e}")
        return

    # Ingestão
    df = carregar_dados_brutos()
    if df is None: return

    # Processamento
    dataset, estatisticas = processar_augmentacao(df, tokenizer)

    # Salvamento JSON (Para a máquina/treinamento)
    with open(ARQUIVO_SAIDA_JSON, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Salvamento CSV (Para humanos conferirem se a separação funcionou)
    df_export = pd.DataFrame(dataset)
    # Removemos colunas complexas para o CSV ficar legível no Excel
    df_export_simple = df_export.drop(columns=['tokens', 'input_ids', 'metadata'])
    df_export_simple['raw_original'] = [d['metadata']['raw_nheengatu'] for d in dataset]
    df_export_simple.to_csv(ARQUIVO_SAIDA_CSV, index=False, encoding='utf-8-sig', sep=';')

    # Relatório Final
    print("\n" + "="*40)
    print("RELATÓRIO DE AUMENTAÇÃO DE DADOS (V2)")
    print("="*40)
    print(f"Linhas Originais (Excel): {estatisticas['original_rows']}")
    print(f"Linhas Geradas (Expandido): {estatisticas['expanded_rows']}")
    print(f"Fator de Multiplicação: {estatisticas['expanded_rows']/estatisticas['original_rows']:.2f}x")
    print(f"Exemplos com [UNK]: {estatisticas['unk_tokens']}")
    print(f"\n✅ Dataset pronto para treino salvo em: {ARQUIVO_SAIDA_JSON}")
    print(f"📊 Tabela para conferência salva em: {ARQUIVO_SAIDA_CSV}")

if __name__ == "__main__":
    main()