import re
import unicodedata

def normalize_unicode(text):
  """
  Converte o texto para a forma normal NFC.
  Isso garante que 'ã' seja um único caractere e não 'a' + '~'.
  """
  if not isinstance(text, str):
    return str(text)
  return unicodedata.normalize('NFC', text)

def clean_text_nheengatu(text):
    """
    Realiza a limpeza profunda do texto preservando a estrutura do Nheengatu.
    
    Etapas:
    1. Normalização Unicode (NFC).
    2. Lowercase (caixa baixa).
    3. Remoção de pontuação (exceto apóstrofo/glotal).
    4. Remoção de espaços extras.
    """

    # 1. Unicode & 2. Lowercase
    text = normalize_unicode(text).lower()

    # 3. Limpeza com Regex
    # A lógica aqui é: Substituir por vazio '' tudo que NÃO for:
    # \w : letras e números (inclui á, ã, ĩ...)
    # \s : espaços
    # '  : o apóstrofo (vital para nhe'eng)
    #
    # O padrão r"[^\w\s']" lê-se: "Qualquer coisa que NÃO seja palavra, espaço ou apóstrofo"
    text = re.sub(r"[^\w\s']", '', text)

    # 4. Remover espaços múltiplos
    # Transforma "ara   puranga" em "ara puranga"
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Bloco de teste rápido (só roda se você executar o arquivo diretamente)
if __name__ == "__main__":
    exemplos = []
    
    print("--- Teste de Normalização ---")
    for original in exemplos:
        limpo = clean_text_nheengatu(original)
        print(f"Original: [{original}]")
        print(f"Limpo:    [{limpo}]")
        print("-" * 30)