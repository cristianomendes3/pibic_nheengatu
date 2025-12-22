import unicodedata

def analyze_string(text, label="Texto"):
    """
    Analisa uma string caractere por caractere, mostrando seu Code Point e Nome.
    """
    print(f"\n--- Analisando: {label} ('{text}') ---")
    print(f"{'Caractere':^10} | {'Code Point':^10} | {'Nome Unicode'}")
    print("-" * 50)
    
    for char in text:
        # ord(char) retorna o número inteiro do Code Point
        # hex(...) converte para hexadecimal (padrão Unicode U+XXXX)
        code_point = hex(ord(char))
        try:
            name = unicodedata.name(char)
        except ValueError:
            name = "<NOME DESCONHECIDO>"
        
        print(f"{char:^10} | {code_point:^10} | {name}")

def normalize_to_nfc(text):
    """
    Força a conversão para a Forma Canônica Composta (NFC).
    Esta é a função que usaremos no pipeline final.
    """
    return unicodedata.normalize('NFC', text)

def normalize_to_nfd(text):
    """
    Força a conversão para a Forma Decomposta (NFD).
    Usada aqui apenas para demonstrar o perigo.
    """
    return unicodedata.normalize('NFD', text)

if __name__ == "__main__":
    # 1. Caracteres Problemáticos e Específicos do Nheengatu/Português
    # Nota: 'ē' (e com macron) é raro em algumas grafias, mas 'ẽ' (e com til) é comum.
    # Vamos testar ambos.
    
    # Caso A: Tils comuns e Cedilha
    palavra_nfc = "maçã" 
    
    # Caso B: Caracteres Indígenas (Nasalização em i e e)
    # 'ĩ' é crucial para o Nheengatu. 'y' é usado como vogal.
    palavra_indigena = "mirĩ" # "Pequeno" em alguns dialetos, ou apenas exemplo de i-til
    
    # Análise 1: Como o Python vê "maçã" digitado normalmente?
    analyze_string(palavra_nfc, "maçã (Original)")
    
    # Análise 2: O Perigo do NFD
    # Vamos forçar o NFD para ver o que acontece "por baixo do capô"
    palavra_nfd = normalize_to_nfd(palavra_nfc)
    analyze_string(palavra_nfd, "maçã (Forçado para NFD)")
    
    # Verificação de Igualdade
    print("\n--- Teste de Comparação ---")
    print(f"Visualmente: '{palavra_nfc}' vs '{palavra_nfd}'")
    print(f"Python diz (==): {palavra_nfc == palavra_nfd}") 
    # ^ Isso deve retornar FALSE, provando que o modelo quebraria sem normalização.
    
    # Análise 3: Caracteres Específicos (ĩ, ē, y)
    # Nota: 'y' sem acento é ASCII puro. 'ỹ' (com til) seria o problemático.
    testes_especiais = "y ĩ ē"
    analyze_string(testes_especiais, "Especiais (y, ĩ, ē)")
    
    # Solução: A Função de Cura
    print("\n--- Aplicando Correção (NFC) ---")
    correcao = normalize_to_nfc(palavra_nfd)
    print(f"Corrigido == Original? {correcao == palavra_nfc}")