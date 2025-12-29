# **Desenvolvimento de Ferramentas Computacionais para o Nheengatu (PIBIC 2025-2026)**
Este repositório contém a infraestrutura técnica desenvolvida na **Fase 1** do projeto, dedicada à implementação da arquitetura computacional desenvolvida para processar, normalizar e analisar vetorialmente a língua **Nheengatu** (Língua Geral Amazônica). O projeto utiliza modelos de Inteligência Artificial (BERT) para auxiliar na documentação de línguas de baixos recursos.

O objetivo central desta fase foi estabelecer um **pipeline de dados robusto** capaz de ingerir planilhas brutas, tratar a instabilidade ortográfica do Nheengatu e validar se modelos de IA atuais conseguem capturar o significado semântico de palavras indígenas.

## **Ciclo de Desenvolvimento da Fase 1:**
O trabalho foi dividido em um ciclo de cinco etapas fundamentais:
1. **Ingestão:** Leitura e conversão de planilhas de campo para formatos otimizados.
2. **Limpeza:** Padronização de textos e tratamento de caracteres especiais.
3. **Processamento (IA):** Uso de modelos de linguagem (BERT) para entender a semântica da língua.
4. **Validação:** Testes científicos para garantir que a máquina está aprendendo corretamente.
5. **Visualização:** Representação gráfica do "mapa mental" (embeddings) criado pelo modelo.

## **Scripts Fundamentais para Reprodução da Pesquisa:**
Para a execução dos primeiros testes com a planilha de 100 palavras fornecida pela equipe de Letras, precisamos passar os dados pelas 5 etapas mencionadas anteriormente. A seguir, executaremos 4 passos fundamentais que realizarão a implementação inicial da arquitetura computacional desenvolvida.

###**Passo 1: Pipeline de Processamento de Dados**
Nesta etapa, iremos operar um pipeline que valida se a arquitetura de ingestão, normalização e tokenização de dados que desenvolvemos é robusta o suficiente para a pesquisa real.

Inicialmente, há um detalhe crucial para a planilha de 100 palavras que recebemos: algumas palavras em Nheengatu possuem mais de um significado, como sinônimos ou flexões de gênero. Além disso, algumas palavras possuem mais de uma grafia por significado. Para primeira análise, fizemos uma alteração manual na planilha, onde utilizamos apenas uma palavra e um significado por linha na primeira versão do pipeline, que chamamos de `run_pipeline.py`.

####**A. Script `run_pipeline.py`:**
Para a primeira versão do pipeline, realizamos três etapas cruciais do ciclo de desenvolvimento, que são: ingestão de dados, normalização e tokenização. Para essas etapas, scripts desenvolvidos anteriormente, como `ingest_data.py` e `normalizer.py` foram integrados ao pipeline, que agora realiza os passos:

1. **Ingestão de Dados:** Carrega a planilha usando pandas.
2. **Normalização:** Aplica lowercase, remove pontuação extra (com exceção do apóstrofo `'`, que é importante para o idioma) e normaliza Unicode (NFC), onde caracteres compostos é tratado como uma única entidade indivisível.
3. **Tokenização:** Aqui, utilizamos um modelo de linguagem BERT pré-treinado para o Nheengatu (acesso em: [canarim-bert-nheengatu](https://huggingface.co/dominguesm/canarim-bert-nheengatu)). O tokenizer servirá como um "dicionário" do modelo. Ele converte texto em números, que utilizará a ferramenta WordPiece para quebrar as palavras em subunidades baseadas na frequência (tokens).
4. **Análise de Qualidade:** Se o modelo encontrar dificuldade para processar caracteres incomuns em Nheengatu, será gerado o token `[UNK]`, e o pipeline identificará irregularidades.

O script `run_pipeline.py` processará a planilha de 100 palavras (com relação de 1 palavra - 1 significado) e irá gerar um arquivo `relatorio_tokens_nheengatu.json`, onde mostrará os tokens "quebrados" que irá dar uma intuição para como a IA está processando a leitura do Nheengatu. Se os tokens ficarem "mais limpos" (menos quebras), saberemos que estamos progredindo.

####**B. Script `pipeline_v2_augment.py`: Evolução do Pipeline para Data Augmentation**
O teste inicial revelou que a relação "1 para 1" (uma palavra, um significado) é insuficiente para representar a realidade linguística do Nheengatu, onde uma palavra pode ter múltiplos significados (ex: *Darápe* = "prato" ou "louça") ou a mesma palavra pode ter escrita de formas diferentes (ex: *Cuára* vs. *Kuara*). Para essa versão, em vez de descartar dados, utilizaremos um algoritmo de expansão. Se uma linha contém 2 variantes da palavra e 2 significados, o script deve gerar **4 exemplos de treinamento** distintos.

Este script substitui o `run_pipeline.py` anterior para fins de criação do dataset mestre (que utilizaremos para próximas análises). Ele integra a normalização profunda e a lógica de expansão. Aqui, é realizado:
1. **Ingestão de Dados Brutos**: com pandas.
2. **Expansão de Linhas (Augmentation):** Regex para separar múltiplos itens e um produto cartesiano: cada variante x cada significado.
3. **Normalização:** lowercase, NFC, remove pontuação exceto glotal.
4. **Tokenização e Validação**

Agora, o script `pipeline_v2_augment.py` processará a planilha bruta de 100 palavras (com múltiplas palavras e múltiplos significados) e expandirá as linhas, em um fator de multiplicação 1.31x, que resultará em um dataset expandido `dataset_nheengatu_expandido.json` com 131 linhas, incluindo palavras, significados, tokens e verificação. Esse dataset agora será utilizado para os passos seguintes.

### **Passo 2: Extração de Inteligência (Embeddings)**
Nesta etapa, iremos passar os dados limpos, processados e expandidos do dataset `dataset_nheengatu_expandido.json` pelo modelo **Canarim-BERT**. Aqui a IA irá "ler" o Nheengatu e converter cada palavra em um vetor numérico de 768 dimensões.

####**C. Script `extraction_script.py`**
Aqui, iremos utilizar dois modelos: Canarim-BERT para o vernáculo indígena e o modelo BERTimbau para o estado da arte em português. No script, iremos carregar e inicializar os modelos para as respectivas línguas e iniciar a extração de embeddings da palavra em Nheengatu e do significado em Português. Eles devolvem uma matriz gigante (768 dimensões) contendo o vetor matemático para cada *token*. Se a palavra foi quebrada em dois tokens pelo modelo, o modelo devolve dois vetores. Para ter um único vetor representando a palavra, calculamos a média desses dois vetores (através do Mean Pooling).

Uma vez executado o script, teremos um arquivo JSON contendo pares de vetores (`embeddings_extraidos.json`). Mas o que esses números nos dizem sobre o "conhecimento" do modelo? Ao realizar a extração de embeddings, conseguimos converter intuições linguísticas (polissemia, sinonímia) em estruturas de dados tangíveis (tensores de ponto flutuante).

É fundamental notar que, após a extração, os vetores de Nheengatu e Português **não habitam o mesmo espaço geométrico.** O vetor no modelo Canarim não significa a mesma coisa que o mesmo vetor no modelo BERTimbau. O script extrai "features" de dois universos paralelos. No entanto, a mineração desses dados fundamentam a próxima grande fase da pesquisa, que envolverá o **Alinhamento de Espaços Vetoriais.** Usaremos os pares extraídos aqui como "pontos de ancoragem" para calcular uma Matriz de Rotação que sobrepõe os dois espaços. Aqui, encontramos a matéria-prima necessária para construir a ponte de tradução automática futura. Sem esses embeddings contextuais precisos, o alinhamento seria ruidoso e a tradução falharia.

###**Passo 3: Validação Matemática**
Nesta etapa, iremos calcular se o vetor da palavra em Nheengatu aponta para a mesma direção do vetor de sua tradução em Português, através da **Similaridade de Cosseno**.

####**D. Script `cosine_validation.py`**
Este script carrega o arquivo `embeddings_extraidos.json`, calcula a similaridade para cada par e gera um relatório estatístico. Aqui, através da função `cosine_similarity` do `sklearn.metrics.pairwise`, iremos gerar valores para cada par Palavra-Significado. Se os cossenos entre os vetores for alto (ex: > 0.5), provamos que os modelos, embora treinados separadamente, compartilham uma estrutura semântica latente compatível ou "alinhável".

No entanto, após calcular as similaridades do arquivo, tivemos um diagnóstico crítico de baixo alinhamento. Com uma Média Geral de Similaridade de 0.0060, Máxima de 0.1034 (*para `'anga' <-> 'alma'`*) e Mínima de -0.0776 (*para `'katuçawa' <-> 'bondade'`*), todos os 131 pares de vetores possuíram uma baixa similaridade (< 0.2). Como estamos usando dois modelos BERT *diferentes* (um treinado em Nheengatu e outro em Português) é normal que a similaridade direta ("zero-shot sem alinhamento") seja baixa (próxima de 0 ou até negativa).

**Por que isso acontece?** Imagine que o modelo Canarim organiza seu "quarto" (espaço vetorial) de um jeito, e o modelo Português organiza o dele de outro. Mesmo que ambos tenham uma "cadeira" (o conceito), ela pode estar no canto esquerdo em um e no direito em outro. Dessa forma, apesar da similaridade baixa, isso não significa fracasso. Significa apenas que precisaremos de uma etapa extra na próxima fase da pesquisa: **aprender uma matriz de rotação que "gira" o espaço do Nheengatu para encaixar no do Português**, mencionado no passo anterior. O teste de similaridade de cosseno serve justamente para diagnosticar a necessidade dessa rotação.

###**Passo 4: Visualização Gráfica dos Resultados**
Nesta etapa, iremos gerar gráficos 2D para que possamos "ver" onde as palavras estão no espaço matemático. Aqui, iremos diagnosticar a qualidade da tokenização através da distribuição de comprimentos e inspecionar a estrutura semântica aprendida pelos modelos através da redução de dimensionalidade.

Para visualizar vetores de 768 dimensões (o output do BERT) em uma tela 2D, precisamos de técnicas matemáticas que preservem a estrutura local (quais palavras são vizinhas) enquanto descartam dimensões menos relevantes. Para essa **Redução de Dimensionalidade**, abordaremos dois métodos:
* **PCA (Principal Component Analysis):** Trata-se de um método linear que rotaciona os eixos dos dados para alinhar com as direções de maior variância. Excelente para ver a estrutura global. Se os vetores de Nheengatu e Português estiverem totalmente separados, isso confirma visualmente o desalinhamento que o cálculo de cosseno indicou.
* **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Um método não-linear e probabilístico. Ele foca em manter vizinhos próximos no espaço original também próximos no espaço reduzido. Ideal para encontrar *clusters* semânticos. Esperamos ver "ilhas" de palavras relacionadas (ex: animais, verbos de movimento) agrupadas.

####**E. Script `visualize_embeddings.py`**
Aqui, o Script gerará três gráficos (salvos na pasta `resultados gráficos`):
1. `distribuicao_tamanho_palavras.png`: Plota a distribuição do tamanho das palavras em caracteres. Ajuda a entender a complexidade morfológica.
   * O Nheengatu é uma língua aglutinante. Para o gráfico gerado, encontramos uma média de caracteres de 5.4. O que não indica uma complexidade morfológica grande para as palavras processadas.

2. `pca_cross_lingual.png`: Plota uma visão global do alinhamento.
   * Para o gráfico gerado, encontramos uma nuvem vermelha (para o Português) e duas nuvens azuis (para o Nheengatu) distantes. As linhas cinzas que ligam as traduções são longas e cruzadas. Isso confirma visualmente que os modelos não falam a mesma "língua matemática" ainda. O *fine-tuning* futuro terá o objetivo de encurtar essas linhas cinzas, puxando os pontos vermelhos para perto dos azuis.

3. `tsne_nheengatu_clusters.png`: Plota a localização geométrica das palavras no espaço, ideal para encontrar *clusters* semânticos.
   * No gráfico gerado, é visível a formação de dois *clusters* separáveis, mas observando as traduções das palavras, o modelo não parece ter capturado a semântica e focado na sua ortografia superficial. Palavras com significados parecidos não parecem estar agrupadas, mas palavras com ortografias parecidas seguem próximas, mesmo com uma separação um pouco "aleatória". Além disso, o modelo aparenta agrupar palavras por classe gramatical (verbos próximos de verbos).

Finalizando a análise da visualização gráfica dos resultados, fechamos o ciclo de `Dados Brutos -> Limpeza -> Processamento (IA) -> Validação -> Visualização`. Dessa forma, completamos a implementação inicial da arquitetura computacional para a planilha de 100 palavras e geramos resultados preliminares sólidos e insights valiosos para fundamentar passos seguintes nas próximas fases da pesquisa!