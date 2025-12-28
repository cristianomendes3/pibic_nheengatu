import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configuração de Estilo
sns.set_theme(style="whitegrid")

INPUT_FILE = "embeddings_extraidos.json"

def load_data():
  try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
      data = json.load(f)
    print(f"✅ Carregados {len(data)} pares de embeddings.")
    return data
  except FileNotFoundError:
    print(f"❌ Arquivo '{INPUT_FILE}' não encontrado. Verifique se o script de extração foi executado.")
    return

def plot_token_distribution(data):
  """
  Plota a distribuição do tamanho das palavras em caracteres.
  Objetivo: Entender a complexidade morfológica.
  """
  words_yrl = [item['nheengatu_text'] for item in data]
  lengths = [len(w) for w in words_yrl]

  plt.figure(figsize=(10, 6))
  sns.histplot(lengths, bins=15, kde=True, color='teal')
  plt.title('Distribuição do Tamanho das Palavras em Nheengatu (Caracteres)')
  plt.xlabel('Tamanho das Palavras (em Caracteres)')
  plt.ylabel('Frequência')
  plt.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Média: {np.mean(lengths):.1f}')
  plt.legend()
  plt.savefig('distribuicao_tamanho_palavras.png')
  print("✅ Gráfico de distribuição salvo.")

def plot_embeddings_2d(data):
  """
  Projeta os embeddings em 2D usando t-SNE e PCA
  """

  # Preparar matrizes
  vecs_yrl = np.array([item['vetor_yrl'] for item in data])
  vecs_pt = np.array([item['vetor_pt'] for item in data])

  # Rótulos para o gráfico
  labels = [item['nheengatu_text'] for item in data]
  # Se futuramente houver categorias no JSON:
  categories = [item.get('categoria', 'Geral') for item in data]


  # 1. PCA (Visão Global do Alinhamento)
  # Concatenamos para ver onde cada língua "mora" no espaço
  all_vecs = np.vstack((vecs_yrl, vecs_pt)) # combina arrays verticalmente, um em cima do outro
  pca = PCA(n_components=2)
  pca_result = pca.fit_transform(all_vecs)

  plt.figure(figsize=(12, 8))
  # Plot Nheengatu points
  plt.scatter(pca_result[:len(data), 0], pca_result[:len(data), 1], 
                c='blue', label='Nheengatu (Canarim)', alpha=0.6)
  # Plot Português points
  plt.scatter(pca_result[len(data):, 0], pca_result[len(data):, 1], 
                c='red', label='Português (BERTimbau)', alpha=0.6)
  
  # Desenhar linhas conectando pares traduzidos (para visualizar a distância)
  for i in range(len(data)):
    start = pca_result[i]      # Ponto YRL
    end = pca_result[len(data)+i] # Ponto PT correspondente
    plt.plot([start[0], end[0]], [start[1], end[1]], color='gray', alpha=0.1)

  plt.title('PCA: Espaços Vetoriais Nheengatu vs Português (Pré-Alinhamento)')
  plt.legend()
  plt.savefig('pca_cross_lingual.png')
  print("✅ Gráfico PCA salvo. Observe se os pontos vermelhos e azuis estão separados (esperado).")

  # 2. t-SNE (Apenas Nheengatu - Análise de Clusters)
  # Vamos ver se palavras similares se agrupam dentro do próprio Nheengatu
  tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='pca', learning_rate='auto')
  tsne_result = tsne.fit_transform(vecs_yrl)

  plt.figure(figsize=(14, 10))
  sns.scatterplot(x=tsne_result[:,0], y=tsne_result[:,1], hue=categories, palette="viridis", s=100)

  # Adicionar anotações de texto para alguns pontos (para não poluir)
  for i, txt in enumerate(labels):
    if i % 2 == 0: # Anota apenas metade para legibilidade
        plt.annotate(txt, (tsne_result[i,0]+0.2, tsne_result[i,1]+0.2), fontsize=9, alpha=0.8)

  plt.title('t-SNE: Mapa Semântico do Nheengatu (Clusters)')
  plt.savefig('tsne_nheengatu_clusters.png')
  print("✅ Gráfico t-SNE salvo. Procure por grupos de palavras com significados próximos.")

if __name__ == "__main__":
  data = load_data()
  if data:
    plot_token_distribution(data)
    plot_embeddings_2d(data)