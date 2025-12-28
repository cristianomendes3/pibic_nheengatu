import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Definição de dispositivos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")

# Mapeamento de Modelos
MODELS_CONFIG = {
    "nheengatu": "dominguesm/canarim-bert-nheengatu",
    "portugues": "neuralmind/bert-base-portuguese-cased"
}

def load_model_and_tokenizer(model_name):
    """
    Carrega o modelo e tokenizador.
    """
    print(f"⏳ Carregando Modelo: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Erro crítico ao carregar modelo: {e}")
        raise e
        
    model.to(device)
    model.eval()
    print(f"✅ Modelo {model_name} carregado!")
    return tokenizer, model

def get_word_embedding(text, target_word, tokenizer, model):
    """
    Extrai o embedding contextual da 'target_word' dentro de 'text'.
    """
    # Se o texto ou a palavra alvo forem nulos, retorna vetor zerado ou trata erro
    if not text or not target_word:
        return np.zeros(768) # Tamanho padrão do BERT

    # Tokenização com offsets para rastrear posições
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
    
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    offset_mapping = encoded["offset_mapping"][0] # Remove dimensão de batch

    # Localizar a palavra no texto (Case insensitive para robustez)
    start_char = text.lower().find(target_word.lower())
    
    if start_char == -1:
        # Fallback: Palavra não encontrada no contexto exato
        return get_isolated_embedding(target_word, tokenizer, model)

    end_char = start_char + len(target_word)

    # Identificar quais tokens correspondem àquela posição de caracteres
    tokens_indices = []
    for idx, (start, end) in enumerate(offset_mapping):
        # Ignora tokens especiais ([CLS], [SEP]) que geralmente têm offset (0,0)
        if start == end: continue 
        
        # Intersecção: Se o token está dentro da faixa da palavra
        # A lógica aqui considera se o token começa ou termina dentro da palavra alvo
        if start >= start_char and end <= end_char:
            tokens_indices.append(idx)

    if not tokens_indices:
        return get_isolated_embedding(target_word, tokenizer, model)

    # Passagem pelo Modelo
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Last Hidden State: (Batch=1, Seq_Len, Hidden=768)
    last_hidden_state = outputs.last_hidden_state[0] # Pega o primeiro item do batch

    # Seleciona os vetores dos tokens encontrados
    # Converter indices para tensor para indexação avançada
    indices_tensor = torch.tensor(tokens_indices, device=device)
    target_vectors = last_hidden_state.index_select(0, indices_tensor)

    # Média dos vetores (Mean Pooling)
    final_embedding = torch.mean(target_vectors, dim=0)

    return final_embedding.cpu().numpy()

def get_isolated_embedding(word, tokenizer, model):
    """Fallback: Extrai embedding da palavra fora de contexto."""
    if not word: return np.zeros(768)
    
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_state = outputs.last_hidden_state[0] # Remove batch: (Seq, Hidden)
    
    # Ignora [CLS] (primeiro) e [SEP] (último) se houver tokens suficientes
    if last_hidden_state.size(0) > 2:
        embedding = torch.mean(last_hidden_state[1:-1], dim=0)
    else:
        embedding = torch.mean(last_hidden_state, dim=0)
        
    return embedding.cpu().numpy()

def main():
    # 1. Carregar o Dataset
    try:
        with open('dataset_nheengatu_expandido.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Erro: dataset_nheengatu_expandido.json não encontrado.")
        return

    # 2. Inicializar Modelos
    try:
        tokenizer_yrl, model_yrl = load_model_and_tokenizer(MODELS_CONFIG['nheengatu'])
        tokenizer_pt, model_pt = load_model_and_tokenizer(MODELS_CONFIG['portugues'])
    except Exception:
        return # Para execução se falhar o load

    results = []

    print("🚀 Iniciando extração de embeddings...")
    for i, item in enumerate(dataset):
        # Log de progresso a cada 10 itens
        if i % 10 == 0: print(f"Processando item {i}/{len(dataset)}...")

        # Usa as chaves corretas do JSON
        word_yrl = item.get('nheengatu_text')
        # Se não houver contexto explícito, usa a própria palavra como contexto
        context_yrl = item.get('nheengatu_text') 
        
        # Extração Nheengatu
        embedding_yrl = get_word_embedding(context_yrl, word_yrl, tokenizer_yrl, model_yrl)

        # Extração Português
        word_pt = item.get('portuguese_text')
        # Se quiser contexto para PT, precisaria estar no JSON. Usando a palavra como fallback.
        context_pt = item.get('portuguese_text')

        embedding_pt = get_word_embedding(context_pt, word_pt, tokenizer_pt, model_pt)

        # Armazenamento
        results.append({
            # Mantém metadados originais se existirem
            "nheengatu_text": word_yrl,
            "portuguese_text": word_pt,
            "metadata": item.get('metadata', {}),
            # Vetores convertidos para lista
            "vetor_yrl": embedding_yrl.tolist(),
            "vetor_pt": embedding_pt.tolist()
        })

    # 3. Salvar Resultados
    output_filename = "embeddings_extraidos.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Sucesso! {len(results)} embeddings salvos em {output_filename}.")

if __name__ == "__main__":
    main()