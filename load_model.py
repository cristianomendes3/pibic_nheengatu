from transformers import AutoTokenizer, AutoModel
import torch

# Nome do modelo no Hugging Face Hub
MODEL_NAME = "dominguesm/canarim-bert-nheengatu"

def load_and_inspect():
  print(f"--- Carregando modelo: {MODEL_NAME} ---")

  # 1. Carregar o Tokenizer
  # O tokenizer é o "dicionário" do modelo. Ele converte texto em números.
  try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Tokenizer carregado com sucesso!")
  except Exception as e:
    print(f"Erro ao carregar o Tokenizer: {e}")
    return

  # 2. Carregar o Modelo (Os "cérebros" da rede neural)
  try:
    model = AutoModel.from_pretrained(MODEL_NAME)
    print(f"Modelo carregado com sucesso!")
    print(f"   - Tamanho do vocabulários: {tokenizer.vocab_size}")
    print(f"   - Arquitetura: {model.config.architectures}")
  except Exception as e:
    print(f"Erro ao carregar o Modelo: {e}")
    return

  # 3. Teste de Tokenização
  # Vamos ver se o modelo entende as raízes do Nheengatu ou se quebra tudo.
  palavras_teste = ["nheengatu", "tata", "paranã", "yauareté", "mba'e", "ara"]
    
  print("\n--- Teste de Tokenização (Morphology Check) ---")
  print(f"{'Palavra':<15} | {'Tokens (Subwords)':<30} | {'IDs'}")
  print("-" * 65)

  for palavra in palavras_teste:
    # Tokeniza a palavra
    tokens = tokenizer.tokenize(palavra)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    # Formatação para visualização
    tokens_str = str(tokens)
    ids_str = str(ids)
    print(f"{palavra:<15} | {tokens_str:<30} | {ids_str}")

  print("\n--- Análise ---")
  print("Se as palavras aparecem inteiras ou com poucas quebras (ex: 'paran', '##ã'),")
  print("o modelo tem um bom vocabulário. Se aparecem letra por letra, é um sinal ruim.")

if __name__ == "__main__":
  load_and_inspect()