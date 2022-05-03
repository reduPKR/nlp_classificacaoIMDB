import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


def gerar_bag_of_words(texto):
    # texto = ["Assisti um filme excelente", "Assisti um filme ótimo","Assisti um filme bom",
    # "Assisti um filme muito bom","Assisti um filme espetacular","Assisti um filme pessimo",
    # "Assisti um filme terrível","Assisti um filme horrível", "Assisti um filme ruim"]

    vetorizar = CountVectorizer(max_features=1000)
    bag_of_words = vetorizar.fit_transform(texto)

    #nao funciona tem que ser matriz esparca
    #matriz = pd.DataFrame(bag_of_words,columns=vetorizar.get_feature_names())

    matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words,columns=vetorizar.get_feature_names_out())
    return matriz_esparsa


def ler_resenha():
    resenha = pd.read_csv("dados/imdb-reviews.csv")

    #padronização de dados normalmente nao esta escrito neg e pos
    classificacao = resenha['sentiment'].replace(["neg","pos"], [0,1])
    resenha["classificacao"] = classificacao
    print(resenha.sentiment.value_counts())
    
    #gerar o bag of words
    bag_of_words = gerar_bag_of_words(resenha.text_pt)

    #gera base de teste e treino do texto X sentmento
    treino, teste, classe_treino, classe_teste = \
        train_test_split(bag_of_words, resenha.classificacao, random_state=50)

    #realiza o treino
    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, classe_treino)

    #mede o quao acurado o modelo está
    acuracia = regressao_logistica.score(teste, classe_teste)
    print(acuracia)


if __name__ == '__main__':
    ler_resenha()