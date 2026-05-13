# recommendation_system

### Visão Geral

Este documento descreve o fluxo completo de um sistema de recomendação híbrido voltado para filmes, construído com dados reais do MovieLens (GroupLens/UMN) e metadados do TMDB, escalável em fases, partindo de filtragem de conteúdo até um modelo híbrido com comportamento real de usuários.

A escolha por filmes é estratégica: o dataset MovieLens já disponibiliza ratings reais de milhões de usuários, resolvendo o problema de cold start do colaborativo desde o dia 1 — algo que a Amazon não oferece publicamente.

### Dados Utilizados

#### Fonte 1: MovieLens (GroupLens / Universidade de Minnesota)

- **URL:** https://grouplens.org/datasets/movielens/
- **Formato:** CSV
- **Licença:** Uso acadêmico e pesquisa gratuito

Três versões disponíveis — escolha conforme a fase do projeto:

| Dataset | Ratings | Filmes | Usuários | Indicado para |
|---|---|---|---|---|
| **MovieLens 100K** | 100.000 | 9.000 | 600 | Protótipo inicial, testes rápidos |
| **MovieLens 1M** | 1.000.000 | 4.000 | 6.000 | Validar modelos, benchmark |
| **MovieLens 25M** | 25.000.000 | 62.000 | 162.000 | Produção, treino de modelos sérios |

### Fonte 2: TMDB (The Movie Database)

- **URL:** https://www.themoviedb.org/documentation/api
- **API gratuita** com cadastro — chave gerada instantaneamente
- Fornece metadados ricos que o MovieLens não tem

### Estrutura dos arquivos MovieLens

**`movies.csv`** — metadados básicos
```csv
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
296,Pulp Fiction (1994),Comedy|Crime|Drama|Thriller
```

**`ratings.csv`** — interações dos usuários
```csv
userId,movieId,rating,timestamp
1,296,5.0,1147880044
1,306,3.5,1147868817
2,296,4.0,1141415820
```

**`tags.csv`** — tags livres inseridas por usuários
```csv
userId,movieId,tag,timestamp
18,4141,Mark Waters,1240597180
65,208,dark comedy,1368150078
65,353,time travel,1418438465
```

**`links.csv`** — ponte para TMDB e IMDb
```csv
movieId,imdbId,tmdbId
1,0114709,862
2,0113497,8844
296,0110912,680
```

### Dados do TMDB (via API ou dataset Kaggle)

A API do TMDB retorna um JSON rico por filme:

```json
{
  "id": 680,
  "title": "Pulp Fiction",
  "original_title": "Pulp Fiction",
  "overview": "A burger-loving hit man, his philosophical partner...",
  "release_date": "1994-10-14",
  "runtime": 154,
  "vote_average": 8.5,
  "vote_count": 25400,
  "popularity": 87.3,
  "genres": [
    {"id": 53, "name": "Thriller"},
    {"id": 80, "name": "Crime"},
    {"id": 35, "name": "Comedy"}
  ],
  "production_countries": [{"iso_3166_1": "US", "name": "United States"}],
  "spoken_languages": [{"iso_639_1": "en", "name": "English"}],
  "keywords": [
    {"id": 925, "name": "drug use"},
    {"id": 4565, "name": "nonlinear timeline"}
  ],
  "credits": {
    "cast": [
      {"name": "John Travolta", "character": "Vincent Vega", "order": 0},
      {"name": "Samuel L. Jackson", "character": "Jules Winnfield", "order": 1},
      {"name": "Uma Thurman", "character": "Mia Wallace", "order": 2}
    ],
    "crew": [
      {"name": "Quentin Tarantino", "job": "Director"},
      {"name": "Lawrence Bender", "job": "Producer"}
    ]
  },
  "poster_path": "/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",
  "backdrop_path": "/suaEOtk1N1sgg2MTM7oZd2cfVp3.jpg"
}
```

### Campos mais importantes para o sistema

| Campo | Fonte | Uso no sistema |
|---|---|---|
| `movieId` | MovieLens | Chave de junção entre todos os arquivos |
| `tmdbId` | links.csv | Buscar metadados ricos no TMDB |
| `userId` | ratings.csv | Identificador do usuário |
| `rating` | ratings.csv | Sinal de preferência (0.5–5.0) |
| `timestamp` | ratings.csv | Ponderação temporal (ratings recentes valem mais) |
| `genres` | movies.csv | Feature categórica principal para conteúdo |
| `tag` | tags.csv | Vocabulário livre dos usuários — rico para TF-IDF |
| `overview` | TMDB | Texto para embeddings semânticos |
| `keywords` | TMDB | Features de conteúdo de alta precisão |
| `cast` (top 3) | TMDB | Feature de elenco |
| `director` | TMDB | Feature de diretor (forte sinal de estilo) |
| `vote_average` | TMDB | Popularidade / fallback para cold start |
| `runtime` | TMDB | Feature numérica auxiliar |

---

## Arquitetura Geral do Sistema

```
┌──────────────────────────────────────────────────────────────┐
│                      CAMADA DE DADOS                         │
│  MovieLens (ratings históricos) + TMDB (metadados de filmes) │
│  + Comportamento dos seus usuários (tempo real)              │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                  CAMADA DE PROCESSAMENTO                     │
│  Limpeza → Feature Engineering → Enriquecimento TMDB         │
└────────────────────────────┬─────────────────────────────────┘
                             │
               ┌─────────────┴─────────────┐
               │                           │
┌──────────────▼──────────┐   ┌────────────▼────────────┐
│   FILTRAGEM DE          │   │   FILTRAGEM              │
│   CONTEÚDO              │   │   COLABORATIVA           │
│   (gêneros, elenco,     │   │   (ratings MovieLens +   │
│   sinopse, keywords)    │   │   eventos dos usuários)  │
└──────────────┬──────────┘   └────────────┬────────────┘
               │                           │
               └─────────────┬─────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                     CAMADA HÍBRIDA                           │
│    score_final = α × score_conteúdo + (1-α) × score_colab   │
│    α adaptativo conforme histórico do usuário                │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                         API REST                             │
│    GET /recommend?user_id=X&limit=10                         │
│    GET /similar?movie_id=X&limit=10                          │
└──────────────────────────────────────────────────────────────┘
```

---

## Fase 1 — Filtragem de Conteúdo

### Conceito

Recomenda filmes similares ao que o usuário já assistiu, baseando-se nos **atributos do filme** — sem precisar de outros usuários.

**Ideal para:** cold start de novos usuários, páginas de "quem assistiu X também pode gostar de Y".

### Como funciona

1. Cada filme é representado como um vetor de features
2. Calcula-se a similaridade entre vetores (cosseno)
3. Retorna os K filmes mais próximos

### Features utilizadas

```python
# Features categóricas (multi-hot — um filme tem vários gêneros)
generos     = ["Crime", "Thriller", "Comedy"]       # de movies.csv
keywords    = ["nonlinear timeline", "drug use"]    # do TMDB

# Features textuais (TF-IDF ou embeddings)
# Combina: sinopse + tags dos usuários + título + keywords como texto
texto       = overview + " " + " ".join(tags) + " " + " ".join(keywords)

# Features de pessoas (tratadas como categorias)
diretor     = "Quentin Tarantino"
elenco_top3 = ["John Travolta", "Samuel L. Jackson", "Uma Thurman"]

# Features numéricas (normalizadas 0–1)
ano_lancamento  = 1994   # normalizar: (ano - 1900) / 130
duracao_min     = 154    # normalizar: duracao / 240
nota_media      = 8.5    # já está em escala 0–10
```

### Pipeline de implementação

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

# 1. Carregar e mesclar dados
movies   = pd.read_csv("movies.csv")        # movieId, title, genres
ratings  = pd.read_csv("ratings.csv")       # userId, movieId, rating, timestamp
tags_df  = pd.read_csv("tags.csv")          # userId, movieId, tag
links    = pd.read_csv("links.csv")         # movieId, imdbId, tmdbId
# tmdb_df = carregar metadados TMDB (JSON enriquecido via API ou dataset Kaggle)

# 2. Processar gêneros (multi-hot)
movies["genres_list"] = movies["genres"].str.split("|")
mlb_genres = MultiLabelBinarizer()
genre_features = mlb_genres.fit_transform(movies["genres_list"])

# 3. Agregar tags por filme como texto
tags_agrupadas = (
    tags_df.groupby("movieId")["tag"]
    .apply(lambda x: " ".join(x.str.lower()))
    .reset_index()
    .rename(columns={"tag": "tags_texto"})
)
movies = movies.merge(tags_agrupadas, on="movieId", how="left")
movies["tags_texto"] = movies["tags_texto"].fillna("")

# 4. Construir corpus de texto (sinopse + tags + título)
# Se tiver TMDB: overview + keywords + tags
movies["corpus"] = (
    movies["title"] + " " +
    movies["tags_texto"]
    # + tmdb_df["overview"] + " " + tmdb_df["keywords_texto"]
)

# 5. TF-IDF no corpus
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)  # captura bigramas como "time travel"
)
text_features = tfidf.fit_transform(movies["corpus"])

# 6. Combinar features com pesos
genre_sparse = csr_matrix(genre_features.astype(float))
features_finais = hstack([
    text_features   * 0.5,   # texto: sinopse + tags
    genre_sparse    * 0.5,   # gêneros
])

# 7. Recomendar filmes similares
def recomendar_similares(movie_id, top_k=10):
    idx = movies.index[movies["movieId"] == movie_id][0]
    item_vec = features_finais[idx]
    scores = cosine_similarity(item_vec, features_finais).flatten()
    indices = scores.argsort()[::-1][1:top_k + 1]
    return movies.iloc[indices][["title", "genres"]].assign(score=scores[indices])
```

### Evolução: Embeddings Semânticos

TF-IDF não entende que "sci-fi" e "science fiction" são a mesma coisa. Embeddings resolvem isso:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # leve: 80MB, 384 dimensões

# Construir texto rico por filme
def construir_texto_filme(row):
    partes = [
        row.get("title", ""),
        row.get("overview", ""),             # sinopse TMDB
        " ".join(row.get("genres_list", [])),
        " ".join(row.get("keywords", [])),   # keywords TMDB
        f"directed by {row.get('director', '')}",
        " ".join(row.get("cast_top3", [])),
        row.get("tags_texto", ""),
    ]
    return " ".join(p for p in partes if p)

movies["texto_rico"] = movies.apply(construir_texto_filme, axis=1)

# Gerar embeddings (feito uma vez, salvo em disco)
embeddings = model.encode(
    movies["texto_rico"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True  # facilita cosseno depois
)
np.save("embeddings_filmes.npy", embeddings)

# Busca eficiente com Faiss (quando o catálogo cresce)
import faiss
index = faiss.IndexFlatIP(embeddings.shape[1])  # produto interno = cosseno (vetores normalizados)
index.add(embeddings.astype("float32"))
```

### Importância do diretor e elenco

Diferente de produtos, em filmes o diretor é um sinal muito forte de estilo:

```python
# Tratar diretor e elenco como "tokens" no corpus
def enriquecer_com_pessoas(row):
    # Substitui espaços por underscore para TF-IDF tratar como token único
    diretor = row.get("director", "").replace(" ", "_")
    elenco = [a.replace(" ", "_") for a in row.get("cast_top3", [])]
    return f"director_{diretor} " + " ".join(f"actor_{a}" for a in elenco)

movies["pessoas"] = movies.apply(enriquecer_com_pessoas, axis=1)
# Adicionar ao corpus antes do TF-IDF
movies["corpus"] += " " + movies["pessoas"]
```

---

## Fase 2 — Filtragem Colaborativa Baseada em Memória

### Conceito

Usa o padrão coletivo de ratings para recomendar. **"Usuários com gostos similares ao seu também gostaram de X."**

O MovieLens já fornece esses ratings — você tem dados colaborativos reais desde o início.

### Quando usar

- Prototipagem rápida (MovieLens 100K é suficiente)
- Base de usuários pequena (< 10k usuários ativos)
- Quando interpretabilidade importa

### Construção da matriz de ratings

```python
from scipy.sparse import csr_matrix

# Mapear IDs para índices inteiros (eficiência de memória)
user_ids  = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_to_idx  = {u: i for i, u in enumerate(user_ids)}
movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

ratings["user_idx"]  = ratings["userId"].map(user_to_idx)
ratings["movie_idx"] = ratings["movieId"].map(movie_to_idx)

# Matriz esparsa (usuários × filmes) — muito mais eficiente que pivot_table
matriz = csr_matrix(
    (ratings["rating"], (ratings["user_idx"], ratings["movie_idx"])),
    shape=(len(user_ids), len(movie_ids))
)
# MovieLens 100K: matriz 600 × 9000 com ~1% de preenchimento (esparsa)
# MovieLens 25M:  matriz 162k × 62k — memória começa a pesar
```

### User-User: "quem se parece com você"

```python
from sklearn.metrics.pairwise import cosine_similarity

def recomendar_user_user(user_id, top_k=10, n_vizinhos=50):
    user_idx = user_to_idx[user_id]
    user_vec = matriz[user_idx]

    # Similaridade com todos os usuários
    sim = cosine_similarity(user_vec, matriz).flatten()
    vizinhos = sim.argsort()[::-1][1:n_vizinhos + 1]

    # Agrega ratings ponderados pela similaridade
    scores = {}
    for idx in vizinhos:
        peso = sim[idx]
        vizinho_vec = matriz[idx].toarray().flatten()
        for movie_idx, rating_val in enumerate(vizinho_vec):
            if rating_val > 0:
                movie_id = movie_ids[movie_idx]
                scores[movie_id] = scores.get(movie_id, 0) + rating_val * peso

    # Remove filmes que o usuário já avaliou
    ja_avaliados = set(ratings[ratings["userId"] == user_id]["movieId"])
    scores = {m: s for m, s in scores.items() if m not in ja_avaliados}

    top_movies = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return movies[movies["movieId"].isin(top_movies)][["title", "genres"]]
```

### Item-Item: "filmes parecidos com o que você já avaliou"

```python
# Transposta: matriz filmes × usuários
sim_filmes = cosine_similarity(matriz.T)

def recomendar_item_item(movie_id, top_k=10):
    movie_idx = movie_to_idx[movie_id]
    scores = sim_filmes[movie_idx]
    indices = scores.argsort()[::-1][1:top_k + 1]
    ids_recomendados = [movie_ids[i] for i in indices]
    return movies[movies["movieId"].isin(ids_recomendados)][["title", "genres"]]
```

### Prós e Contras

| Prós | Contras |
|---|---|
| Simples de implementar | Não escala acima de ~10k usuários/filmes |
| Sem fase de treino | Recalcular similaridade em tempo real é lento |
| Atualiza imediatamente com novos dados | Sofre com matrizes esparsas |
| Interpretável para o usuário | Cold start severo para novos usuários |

---

## Fase 3 — Filtragem Colaborativa Baseada em Modelo

### Conceito: Matrix Factorization

Decompõe a matriz de ratings em dois conjuntos de vetores latentes:

```
R (usuários × filmes) ≈ U (usuários × k) × Vᵀ (k × filmes)
```

O modelo aprende que "este usuário prefere filmes de Nolan dos anos 2000 com narrativas não-lineares" sem você definir isso. Cada dimensão latente captura um padrão escondido.

### SVD — para ratings explícitos (ideal para MovieLens)

O MovieLens tem ratings de 0.5 a 5.0 — dados explícitos, perfeito para SVD:

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

# Carregar dataset no formato Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

# Validação cruzada para escolher hiperparâmetros
model = SVD(
    n_factors=100,    # dimensões dos vetores latentes
    n_epochs=30,      # passagens pelo dataset
    lr_all=0.005,     # taxa de aprendizado
    reg_all=0.02      # regularização (evita overfitting)
)
resultados = cross_validate(model, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
# MovieLens 100K: RMSE ~0.87 | MovieLens 1M: RMSE ~0.85

# Treinar no dataset completo
trainset = data.build_full_trainset()
model.fit(trainset)

# Prever rating de um usuário para um filme específico
predicao = model.predict(uid=1, iid=296)  # userId=1, movieId=296 (Pulp Fiction)
print(f"Rating estimado: {predicao.est:.2f}")

# Gerar top-N recomendações para um usuário
def top_n_svd(user_id, n=10):
    filmes_avaliados = set(ratings[ratings["userId"] == user_id]["movieId"])
    filmes_candidatos = [m for m in movie_ids if m not in filmes_avaliados]

    predicoes = [(m, model.predict(user_id, m).est) for m in filmes_candidatos]
    predicoes.sort(key=lambda x: x[1], reverse=True)

    top_ids = [m for m, _ in predicoes[:n]]
    return movies[movies["movieId"].isin(top_ids)][["title", "genres"]]
```

### ALS — para dados implícitos (quando você coleta comportamento)

Quando os usuários da sua plataforma não dão ratings explícitos, mas você coleta cliques e tempo de visualização:

```python
from implicit import als
import scipy.sparse as sp

# Construir matriz de confiança (confidence matrix)
# Dados implícitos: quanto mais interações, maior a confiança
def eventos_para_confianca(eventos_df, alpha=40):
    """
    alpha: fator de escala. confidence = 1 + alpha * interacoes
    Eventos: 1 clique = 1, assistiu 30min = 3, assistiu completo = 8
    """
    confianca = eventos_df.groupby(["user_idx", "movie_idx"])["peso"].sum()
    conf_matriz = sp.csr_matrix(
        (1 + alpha * confianca.values, (confianca.index.get_level_values(0),
                                        confianca.index.get_level_values(1))),
        shape=(n_users, n_movies)
    )
    return conf_matriz

model_als = als.AlternatingLeastSquares(
    factors=128,
    regularization=0.01,
    iterations=30,
    use_gpu=False
)

conf_matrix = eventos_para_confianca(eventos_df)
model_als.fit(conf_matrix)  # espera matriz item × usuário
ids, scores = model_als.recommend(user_idx, conf_matrix[user_idx], N=10)
```

### Neural Collaborative Filtering (NCF)

Quando SVD e ALS não são suficientes — captura interações não-lineares:

```python
import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, n_users, n_movies, embed_dim=64, layers=[128, 64, 32]):
        super().__init__()
        self.user_embed  = nn.Embedding(n_users, embed_dim)
        self.movie_embed = nn.Embedding(n_movies, embed_dim)

        mlp_layers = []
        input_dim = embed_dim * 2
        for out_dim in layers:
            mlp_layers += [nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Dropout(0.2)]
            input_dim = out_dim
        mlp_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, user_ids, movie_ids):
        u = self.user_embed(user_ids)
        v = self.movie_embed(movie_ids)
        x = torch.cat([u, v], dim=-1)
        return self.mlp(x).squeeze()
```

### Comparativo dos modelos

| Modelo | Dados ideais | Escala | Complexidade | Quando usar |
|---|---|---|---|---|
| **SVD** | Ratings explícitos (0.5–5.0) | Alta | Baixa | Dataset MovieLens, protótipo sólido |
| **ALS** | Implícitos (cliques, tempo visto) | Alta | Média | Plataforma com usuários reais sem rating |
| **NCF** | Ambos | Muito alta | Alta | Quando SVD/ALS atingirem teto de qualidade |

---

## Fase 4 — Modelo Híbrido

### Conceito

Combina os scores dos dois sistemas. A proporção `α` é ajustada de acordo com o quanto você conhece o usuário.

```
score_final = α × score_conteúdo + (1 - α) × score_colaborativo
```

### Estratégia de α adaptativo

```python
def calcular_alpha(n_ratings_usuario):
    """
    Usuário novo (poucos ratings) → confia mais no conteúdo
    Usuário experiente (muitos ratings) → confia mais no colaborativo
    
    Referência MovieLens: usuário mediano tem ~165 ratings
    """
    if n_ratings_usuario < 5:
        return 0.9    # quase tudo conteúdo (cold start)
    elif n_ratings_usuario < 20:
        return 0.65
    elif n_ratings_usuario < 50:
        return 0.35
    else:
        return 0.1    # quase tudo colaborativo

def recomendar_hibrido(user_id, n=10):
    n_ratings = len(ratings[ratings["userId"] == user_id])
    alpha = calcular_alpha(n_ratings)

    # Candidatos: união dos top-50 de cada modelo
    candidatos_conteudo  = top_n_conteudo(user_id, n=50)
    candidatos_colab     = top_n_svd(user_id, n=50)
    candidatos           = list(set(candidatos_conteudo) | set(candidatos_colab))

    scores_finais = {}
    for movie_id in candidatos:
        score_c   = score_conteudo(user_id, movie_id)   # similaridade com histórico
        score_col = model.predict(user_id, movie_id).est / 5.0  # normaliza 0–1

        scores_finais[movie_id] = alpha * score_c + (1 - alpha) * score_col

    top_ids = sorted(scores_finais, key=scores_finais.get, reverse=True)[:n]
    return movies[movies["movieId"].isin(top_ids)][["title", "genres"]]
```

### Outras estratégias de combinação

- **Switching:** usa conteúdo se `n_ratings < 10`, senão colaborativo — mais simples, menos suave
- **Cascade:** colaborativo gera 100 candidatos, conteúdo re-rankeia com base no perfil de gênero do usuário
- **Feature augmentation:** embeddings do conteúdo do filme entram como features no modelo colaborativo neural

### Vantagem específica de filmes para o híbrido

Filmes têm atributos muito mais ricos que produtos genéricos. O conteúdo consegue fazer recomendações de qualidade mesmo com pouquíssimo histórico, porque gênero + diretor + elenco são sinais fortes de preferência.

---

## Coleta de Comportamento dos Usuários

### Eventos a registrar

Para uma plataforma de filmes, os eventos relevantes são diferentes de um e-commerce:

```python
# Tabela de eventos
{
    "event_id": "uuid",
    "user_id": "user_abc",
    "movie_id": 296,
    "event_type": "impression"        # apareceu na tela
               | "click"             # clicou para ver detalhes
               | "trailer_play"      # assistiu ao trailer
               | "watchlist_add"     # adicionou à lista
               | "watch_start"       # começou a assistir
               | "watch_complete"    # assistiu >85% do filme
               | "watch_abandon"     # parou antes de 30%
               | "rating"            # deu uma nota (0.5–5)
               | "share",            # compartilhou
    "value": 4.5,                    # para rating; null para outros
    "watch_pct": 0.92,               # % do filme assistido
    "timestamp": 1690000000,
    "source": "recommendation" | "search" | "homepage" | "direct"
}
```

### Pesos por tipo de evento (dados implícitos)

| Evento | Peso | Raciocínio |
|---|---|---|
| `impression` | 0.1 | Viu mas não interagiu — sinal fraco |
| `click` | 1.0 | Interesse real na sinopse |
| `trailer_play` | 2.0 | Intenção clara |
| `watchlist_add` | 3.0 | Intenção de assistir |
| `watch_start` | 3.0 | Começou a assistir |
| `watch_complete` | 8.0 | Sinal mais forte possível |
| `watch_abandon` | -1.0 | Sinal negativo — não gostou |
| `rating` (1–5) | `rating × 2` | Feedback explícito, muito valioso |
| `share` | 5.0 | Recomendou para outra pessoa — altíssima aprovação |

### Conversão para matrix de preferências

```python
def eventos_para_matriz(eventos_df):
    pesos = {
        "impression": 0.1, "click": 1.0, "trailer_play": 2.0,
        "watchlist_add": 3.0, "watch_start": 3.0,
        "watch_complete": 8.0, "watch_abandon": -1.0, "share": 5.0
    }
    eventos_df["peso_base"] = eventos_df["event_type"].map(pesos)

    # Ajuste por porcentagem assistida (para watch_start)
    mask_watch = eventos_df["event_type"] == "watch_start"
    eventos_df.loc[mask_watch, "peso_base"] *= eventos_df.loc[mask_watch, "watch_pct"].fillna(0.5)

    # Rating explícito sobrescreve implícito
    mask_rating = eventos_df["event_type"] == "rating"
    eventos_df.loc[mask_rating, "peso_base"] = eventos_df.loc[mask_rating, "value"] * 2

    # Agrega por usuário/filme
    return (
        eventos_df.groupby(["user_id", "movie_id"])["peso_base"]
        .sum()
        .clip(lower=0)   # score mínimo 0
        .reset_index()
        .rename(columns={"peso_base": "preferencia"})
    )
```

---

## Escalabilidade

### Estratégias por fase de crescimento

**Fase inicial (< 10k usuários / catálogo ~10k filmes):**
- Tudo em memória com pandas + scikit-learn
- Matriz de similaridade calculada em batch noturno
- SVD treinado semanalmente com dataset completo
- SQLite ou PostgreSQL para eventos
- API com FastAPI

**Fase intermediária (10k–500k usuários):**
- Pré-computa recomendações para cada usuário e salva no banco
- Redis para cache dos top-N de cada usuário (TTL de 24h)
- Treino ALS/SVD agendado com Celery
- Faiss para busca de filmes similares por embedding

```python
import faiss
import numpy as np

# Indexar embeddings para busca aproximada eficiente
embeddings = np.load("embeddings_filmes.npy").astype("float32")
d = embeddings.shape[1]  # 384 (all-MiniLM) ou 768 (MPNet)

# IVF = busca por partições (mais rápido que força bruta)
n_cells = 100  # número de partições — sqrt(n_filmes) é uma boa referência
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, n_cells, faiss.METRIC_INNER_PRODUCT)

index.train(embeddings)
index.add(embeddings)
index.nprobe = 10  # quantas partições checar na busca

# Busca: top 10 similares para um filme
D, I = index.search(embeddings[[filme_idx]], k=11)  # k+1 pois inclui o próprio
filmes_similares = movie_ids[I[0][1:]]
```

**Fase avançada (> 500k usuários):**
- Two-Tower model (rede neural separada para usuário e filme)
- Feature store com Feast para features em tempo real
- Pipeline de retreino contínuo: Kafka → Spark → treino → deploy
- Serving com TorchServe ou TensorFlow Serving

---

## Avaliação do Sistema

### Métricas offline (com dataset histórico)

```python
from sklearn.metrics import mean_squared_error, ndcg_score
import numpy as np

# RMSE — erro de predição de rating (baseline MovieLens: ~0.87)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Precision@K — dos K recomendados, quantos são relevantes
# (relevante = rating >= 4.0)
def precision_at_k(recomendados, relevantes, k=10):
    return len(set(recomendados[:k]) & set(relevantes)) / k

# Recall@K — dos filmes que o usuário gostaria, quantos foram recomendados
def recall_at_k(recomendados, relevantes, k=10):
    if not relevantes:
        return 0.0
    return len(set(recomendados[:k]) & set(relevantes)) / len(relevantes)

# NDCG@K — penaliza recomendações relevantes em posições ruins
ndcg = ndcg_score([relevances_true], [relevances_predicted], k=10)

# Hit Rate@K — em quantas sessões a recomendação acertou ao menos 1 filme
def hit_rate_at_k(recomendados_todos, relevantes_todos, k=10):
    hits = sum(
        1 for rec, rel in zip(recomendados_todos, relevantes_todos)
        if set(rec[:k]) & set(rel)
    )
    return hits / len(recomendados_todos)
```

### Métricas online (com usuários reais)

| Métrica | Como medir | O que indica |
|---|---|---|
| **CTR** | cliques / impressões | Relevância percebida |
| **Watch Rate** | watch_complete / click | Qualidade real da recomendação |
| **Rating após recomendação** | média de ratings em filmes recomendados | Satisfação medida |
| **Coverage** | % do catálogo recomendado | Diversidade (evita bolha de popularidade) |
| **Novelty** | quão desconhecidos são os filmes recomendados | Capacidade de descoberta |
| **Serendipity** | filmes fora do padrão mas bem avaliados | Satisfação a longo prazo |

### Baseline mínimo para comparar

Antes de qualquer modelo, implemente um baseline simples:

```python
# Baseline 1: popularidade global (filmes mais bem avaliados)
populares = (
    ratings.groupby("movieId")["rating"]
    .agg(["mean", "count"])
    .query("count >= 50")           # mínimo de ratings para confiabilidade
    .sort_values("mean", ascending=False)
    .head(10)
)

# Baseline 2: bayesian average (penaliza filmes com poucos ratings)
C = ratings["rating"].mean()       # rating médio global
m = 50                             # mínimo de ratings para confiança

def bayesian_avg(grupo):
    v = len(grupo)
    R = grupo.mean()
    return (v / (v + m)) * R + (m / (v + m)) * C

baseline_bayesian = ratings.groupby("movieId")["rating"].apply(bayesian_avg)
```

Qualquer modelo deve superar esses baselines. Se não superar, há algo errado.

---

## Stack Tecnológica Recomendada

```
Linguagem:        Python 3.11+
API:              FastAPI
Banco de dados:   PostgreSQL (eventos) + Redis (cache de recomendações)
ML colaborativo:  scikit-surprise (SVD), implicit (ALS), PyTorch (NCF)
ML conteúdo:      scikit-learn (TF-IDF), sentence-transformers (embeddings)
Busca vetorial:   Faiss
Dados de filmes:  MovieLens (ratings), TMDB API (metadados)
Orquestração:     Celery + Redis (jobs de retreino agendado)
Containerização:  Docker + Docker Compose
Monitoramento:    Prometheus + Grafana (métricas de modelo e negócio)
```

---

## Roadmap de Implementação

```
Semana 1–2
└── Download MovieLens 100K + limpeza dos CSVs
└── Enriquecimento com TMDB (gêneros, sinopse, diretor, elenco)
└── Pipeline de filtragem de conteúdo (TF-IDF + similaridade cosseno)
└── Endpoint GET /similar?movie_id=X

Semana 3–4
└── Construção da matriz de ratings (userId × movieId)
└── Colaborativo baseado em memória (item-item)
└── Endpoint GET /recommend?user_id=X
└── Comparar com baselines de popularidade

Semana 5–6
└── Treinar SVD com scikit-surprise no MovieLens completo
└── Validação cruzada: medir RMSE, Precision@10, NDCG@10
└── Substituir colaborativo em memória pelo SVD

Semana 7–8
└── Implementar α adaptativo e combinar conteúdo + SVD
└── Adicionar coleta de eventos dos usuários reais
└── Cache Redis para top-N pré-computados por usuário

Semana 9–10
└── Substituir TF-IDF por embeddings (sentence-transformers + Faiss)
└── Medir impacto nos KPIs online (CTR, Watch Rate)
└── Retreino agendado com dados próprios mesclados ao MovieLens

Semana 11+
└── Monitorar métricas online e drift do modelo
└── Evoluir para ALS se coletar dados implícitos suficientes
└── Considerar NCF ou Two-Tower quando SVD atingir teto
```

---

## Resumo dos Conceitos-Chave

| Conceito | Descrição | Quando entra |
|---|---|---|
| **TF-IDF** | Peso de palavras por frequência inversa — bom para sinopses e tags | Fase 1 |
| **Multi-hot encoding** | Vetor binário de gêneros — um filme pode ter vários | Fase 1 |
| **Embedding semântico** | Vetor denso de significado (sentence-transformers) — entende sinônimos | Fase 1 evolução |
| **Similaridade cosseno** | Ângulo entre vetores, independente de magnitude | Fases 1 e 2 |
| **Esparsidade** | Poucos ratings em relação ao total possível (matriz >99% vazia) | Problema central |
| **Cold start** | Usuário/filme sem histórico suficiente — resolvido pelo híbrido | Todo o sistema |
| **Fatores latentes** | Dimensões ocultas aprendidas pelo SVD/ALS (ex: "gosta de thrillers dos anos 90") | Fase 3 |
| **Rating explícito** | Nota que o usuário dá (0.5–5.0) — dados do MovieLens | Fase 3 SVD |
| **Dado implícito** | Comportamento inferido (% assistido, clique) — seus usuários reais | Fase 3 ALS |
| **α adaptativo** | Peso dinâmico entre conteúdo e colaborativo conforme histórico cresce | Fase 4 |
| **Bayesian average** | Penaliza filmes com poucos ratings — evita que filmes raros pareçam ótimos | Avaliação |
| **Faiss** | Busca eficiente de vizinhos mais próximos em vetores de embedding | Escala |
| **NDCG** | Métrica que penaliza recomendações relevantes mal posicionadas no ranking | Avaliação |
| **Watch Rate** | % que assistiu ao filme após a recomendação — melhor KPI online para filmes | Online |
