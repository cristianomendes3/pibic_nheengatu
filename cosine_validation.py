import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Configuração de Entrada
INPUT_FILE = "embeddings_extraidos.json"
OUTPUT_REPORT = "relatorio_similaridade_cosseno.csv"

def load_embeddings(filename):
    """Carrega o JSON e converte listas de volta para arrays numpy."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Carregados {len(data)} pares de embeddings.")
        return data
    except FileNotFoundError:
        print(f"❌ Arquivo '{filename}' não encontrado. Verifique se o script de extração foi executado.")
        return []
    except json.JSONDecodeError:
        print(f"❌ Erro ao decodificar o arquivo JSON '{filename}'.")
        return []

def calculate_similarities(data):
    """Calcula a similaridade de cosseno para cada par Nheengatu-Português."""
    results = []
    
    print("--- Calculando Similaridades ---")
    
    for item in data:
        # Recupera vetores e garante que são arrays 2D (1, 768)
        vec_yrl = np.array(item['vetor_yrl']).reshape(1, -1)
        vec_pt = np.array(item['vetor_pt']).reshape(1, -1)
        
        # Calcula Cosseno
        # O sklearn retorna uma matriz [[score]], pegamos o valor escalar com [0][0]
        similarity = cosine_similarity(vec_yrl, vec_pt)[0][0]
        
        # Armazena resultado usando as chaves corretas do novo JSON
        results.append({
            "Nheengatu": item.get('nheengatu_text', 'N/A'),
            "Portugues": item.get('portuguese_text', 'N/A'),
            # Metadados opcionais
            "Fonte": item.get('metadata', {}).get('raw_nheengatu', 'N/A'),
            "Similaridade": float(similarity) # Garante que é um float Python puro
        })
        
    return results

def analyze_results(results):
    """Gera estatísticas descritivas dos resultados."""
    if not results:
        print("Nenhum resultado para analisar.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    
    print("\n" + "="*40)
    print("RELATÓRIO DE VALIDAÇÃO CROSS-LINGUAL")
    print("="*40)
    
    # Estatísticas Específicas da Coluna de Similaridade
    sim_series = df['Similaridade']
    
    mean_sim = sim_series.mean()
    max_sim = sim_series.max()
    min_sim = sim_series.min()
    
    # Identificar os pares de maior e menor similaridade
    best_pair = df.loc[sim_series.idxmax()]
    worst_pair = df.loc[sim_series.idxmin()]

    print(f"Média Geral de Similaridade: {mean_sim:.4f}")
    print(f"Máxima: {max_sim:.4f} ('{best_pair['Nheengatu']}' <-> '{best_pair['Portugues']}')")
    print(f"Mínima: {min_sim:.4f} ('{worst_pair['Nheengatu']}' <-> '{worst_pair['Portugues']}')")
    
    # Análise por Faixas
    high_conf = len(df[df['Similaridade'] > 0.5])
    low_conf = len(df[df['Similaridade'] < 0.2])
    total = len(df)
    
    print(f"\nPares com Alta Similaridade (> 0.5): {high_conf} ({(high_conf/total)*100:.1f}%)")
    print(f"Pares com Baixa Similaridade (< 0.2): {low_conf} ({(low_conf/total)*100:.1f}%)")
    
    # Diagnóstico Interpretativo
    print("\n--- Diagnóstico ---")
    if mean_sim > 0.5:
        print("✅ SUCESSO: O alinhamento cross-lingual é forte.")
        print("   O modelo Canarim já possui boa correspondência com o português.")
    elif mean_sim > 0.3:
        print("⚠️ ATENÇÃO: Alinhamento moderado.")
        print("   Existe correspondência, mas ruídos de tokenização ou polissemia")
        print("   podem estar interferindo. Pode ser necessário fine-tuning.")
    else:
        print("❌ CRÍTICO: Baixo alinhamento.")
        print("   Os espaços vetoriais parecem distantes. Isso é comum se os modelos")
        print("   não foram treinados como bilíngues pareados. Considere treinar")
        print("   uma matriz de projeção linear (Orthogonal Procrustes).")

    return df

def main():
    data = load_embeddings(INPUT_FILE)
    if not data: return
    
    results = calculate_similarities(data)
    df = analyze_results(results)
    
    if not df.empty:
        # Salvar CSV para inspeção humana
        df.to_csv(OUTPUT_REPORT, index=False, encoding='utf-8-sig', sep=';', float_format='%.4f')
        print(f"\n📄 Relatório detalhado salvo em: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()