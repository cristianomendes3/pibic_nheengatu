from transformers import AutoTokenizer

def analyze_tokens():
  # 1. Carregar os Tokenizers
  print("--- Carregando Tokenizers... ---")

  # Modelo Especializado (Baseado em BERT/WordPiece)
  try:
    tokenizer_canarim = AutoTokenizer.from_pretrained("dominguesm/canarim-bert-nheengatu")
    print(f"Tokenizer Canarim (WordPiece) carregado!")
  except Exception as e:
    print(f"Erro ao carregar o Modelo Canarim: {e}")
    return
  
  # Modelo Generalista (Baseado em XLM-R/SentencePiece)
  try:
    tokenizer_xlmr = AutoTokenizer.from_pretrained("xlm-roberta-base")
    print(f"Tokenizer XLM-R (SentencePiece) carregado!")
  except Exception as e:
    print(f"Erro ao carregar o Modelo XLM-R: {e}")
    return

  # 2. Lista de Palavras Complexas ("Torture Test")
  # Inclui: Glotais ('), Tils (ã, ẽ), Hifens (-), e palavras aglutinadas
  palavras_teste = ["tukũ", "yapukuĩ", "nhe'eng", "yauareté", "kĩdara", "çēdú", "Kuyera imiráwara"]

  print("\n" + "="*80)
  print(f"{'Palavra Original':<15} | {'Canarim (Especializado)':<30} | {'XLM-R (Generalista)'}")
  print("="*80)

  for palavra in palavras_teste:
    # Tokenização Canarim
    tokens_can = tokenizer_canarim.tokenize(palavra)
    # Tokenização XLM-R
    tokens_xlm = tokenizer_xlmr.tokenize(palavra)

    print(f"{palavra:<15} | {str(tokens_can):<30} | {str(tokens_xlm)}")

  print("="*80)
  print("\n")
  print("1. Observe como 'nhe'eng' foi quebrado. O apóstrofo sumiu ou virou um token separado?")
  print("2. O Canarim usa '##' para sufixos. O XLM-R usa ' ' para inícios.")
  print("3. Palavras com muitos pedaços pequenos indicam que o modelo 'não conhece' a palavra.")

if __name__ == "__main__":
    analyze_tokens()