import pandas as pd
import numpy as np

def calc_par_pertinencia(limites_pertinencia):
    # define os parâmetros das funções triangulares baseando-se nos limites dos bins do histograma
    parametros_pertinencia = pd.DataFrame(columns=["a", "b", "c"])
    parametros_pertinencia.loc["cai_muito", "b"] = limites_pertinencia[0]-(-limites_pertinencia[0]+limites_pertinencia[1])/2
    parametros_pertinencia.loc["cai_muito", "c"] = (limites_pertinencia[0]+limites_pertinencia[1])/2
    parametros_pertinencia.loc["cai_pouco", "a"] = parametros_pertinencia.loc["cai_muito", "b"]
    parametros_pertinencia.loc["cai_pouco", "b"] = parametros_pertinencia.loc["cai_muito", "c"]
    parametros_pertinencia.loc["cai_pouco", "c"] = (limites_pertinencia[1]+limites_pertinencia[2])/2
    parametros_pertinencia.loc["estavel", "a"] = parametros_pertinencia.loc["cai_pouco", "b"]
    parametros_pertinencia.loc["estavel", "b"] = parametros_pertinencia.loc["cai_pouco", "c"]
    parametros_pertinencia.loc["estavel", "c"] = (limites_pertinencia[2]+limites_pertinencia[3])/2
    parametros_pertinencia.loc["sobe_pouco", "a"] = parametros_pertinencia.loc["estavel", "b"]
    parametros_pertinencia.loc["sobe_pouco", "b"] = parametros_pertinencia.loc["estavel", "c"]
    parametros_pertinencia.loc["sobe_pouco", "c"] = limites_pertinencia[3]+(limites_pertinencia[3]-limites_pertinencia[2])/2
    parametros_pertinencia.loc["sobe_muito", "a"] = parametros_pertinencia.loc["sobe_pouco", "b"]
    parametros_pertinencia.loc["sobe_muito", "b"] = parametros_pertinencia.loc["sobe_pouco", "c"]# define os parâmetros das funções triangulares baseando-se nos limites dos bins do histograma
    parametros_pertinencia = pd.DataFrame(columns=["a", "b", "c"])
    parametros_pertinencia.loc["cai_muito", "b"] = limites_pertinencia[0]-(-limites_pertinencia[0]+limites_pertinencia[1])/2
    parametros_pertinencia.loc["cai_muito", "c"] = (limites_pertinencia[0]+limites_pertinencia[1])/2
    parametros_pertinencia.loc["cai_pouco", "a"] = parametros_pertinencia.loc["cai_muito", "b"]
    parametros_pertinencia.loc["cai_pouco", "b"] = parametros_pertinencia.loc["cai_muito", "c"]
    parametros_pertinencia.loc["cai_pouco", "c"] = (limites_pertinencia[1]+limites_pertinencia[2])/2
    parametros_pertinencia.loc["estavel", "a"] = parametros_pertinencia.loc["cai_pouco", "b"]
    parametros_pertinencia.loc["estavel", "b"] = parametros_pertinencia.loc["cai_pouco", "c"]
    parametros_pertinencia.loc["estavel", "c"] = (limites_pertinencia[2]+limites_pertinencia[3])/2
    parametros_pertinencia.loc["sobe_pouco", "a"] = parametros_pertinencia.loc["estavel", "b"]
    parametros_pertinencia.loc["sobe_pouco", "b"] = parametros_pertinencia.loc["estavel", "c"]
    parametros_pertinencia.loc["sobe_pouco", "c"] = limites_pertinencia[3]+(limites_pertinencia[3]-limites_pertinencia[2])/2
    parametros_pertinencia.loc["sobe_muito", "a"] = parametros_pertinencia.loc["sobe_pouco", "b"]
    parametros_pertinencia.loc["sobe_muito", "b"] = parametros_pertinencia.loc["sobe_pouco", "c"]

    return parametros_pertinencia

def calc_pertinencia(x, pertinencia):
    """Função que retorna a pertinencia de cada ponto x a cada combinação de pertinecia utilizada. Recebe como
    entrada os dados, os parametros de pertinência e retorna os valores de pertinencia para cada x
    de cada regra.
    return -> dataframe[x][xn, pertinencia]
    """
    pertinencia = pertinencia / 100
    index = pd.MultiIndex.from_product((x.columns, pertinencia.index))
    resultado = pd.DataFrame(columns=index, index=range(len(x)))

    for moeda in x:
        for i_pert in pertinencia.index:
            # caso o valor de x seja superior a b, aplica a função decrescente entre os pontos b e c.
            indices = np.where(x.loc[:, moeda] >= pertinencia.loc[i_pert, "b"])[0]
            if np.isnan(pertinencia.loc[i_pert, "c"]):
                resultado.loc[indices, (moeda, i_pert)] = 1
            else:
                resultado.loc[indices, (moeda, i_pert)] = (
                (x.loc[x.index[indices], moeda] - pertinencia.loc[i_pert, "c"]) /
                (pertinencia.loc[i_pert, "b"] - pertinencia.loc[i_pert, "c"])).values

            # caso os valores de x sejam inferiores a b, aplica a função crescente entre os pontos a e b.
            indices = np.where(x.loc[:, moeda] < pertinencia.loc[i_pert, "b"])[0]
            if np.isnan(pertinencia.loc[i_pert, "a"]):
                resultado.loc[indices, (moeda, i_pert)] = 1
            else:
                resultado.loc[indices, (moeda, i_pert)] = (
                (x.loc[x.index[indices], moeda] - pertinencia.loc[i_pert, "a"]) /
                (pertinencia.loc[i_pert, "b"] - pertinencia.loc[i_pert, "a"])).values

            # define o valor 0 para pertinências negativas
            indices = np.where(resultado.loc[:, (moeda, i_pert)] < 0)[0];
            resultado.loc[indices, (moeda, i_pert)] = 0

    resultado.index = x.index

    return resultado

def classifica_saida(dados,vender,manter,classes):
    saida_classificada = pd.DataFrame(index=dados.index, columns=[dados.columns[-1]])
    indices = np.where(dados.iloc[:,-1] <= vender/100) [0]
    saida_classificada.iloc[indices,:] = str(classes[0])
    indices = np.where((dados.iloc[:,-1] > vender/100) & (dados.iloc[:,-1] < manter/100)) [0]
    saida_classificada.iloc[indices,:] = str(classes[1])
    indices = np.where(dados.iloc[:,-1] >= manter/100) [0]
    saida_classificada.iloc[indices,:] = str(classes[2])
    
    return saida_classificada


def calcula_uj(matriz_pertinencias, individuo):
    """Função que gera o dataframe uj, correspondente às pertinencias para o antecedente de cada regra para um dado
    indivíduo. Recebe como entrada a matriz de pertinencia dos dados e a lista de regras do individuo. Utiliza o produto
    como T-norma.
    Retorna o dataframe uj onde cada coluna representa uma regra do indivíduo.
    """
    
    uj = pd.DataFrame(columns=range(len(individuo)),index=matriz_pertinencias.index)
    uj.loc[:,:] = 1
    for i in range(len(individuo)):
        for j in range(round(len(individuo[i])/2)):
            uj.loc[:,i] = uj.loc[:,i] * matriz_pertinencias.loc[:,(individuo[i][j*2],individuo[i][(j*2)+1])]

    return uj;

def realiza_treinamento(dados, saida_classificada, uj, individuo, classes):
    """Função que realiza o treinamento para uma base de dados de entradas x e saida_classificada. Recebe como entrada 
    os dados, o vetor saida saida_classificada, as pertinencias às regras uj, o indivíduo analisado e as classes existentes.
    Retorna um dataframe contendo o resultado das classificações.
    """ 
    x = dados.copy(deep=True)
    x = x.drop(x.columns[-1], axis=1)
    
    uj = uj[uj.index.isin(dados.index)]
    saida_classificada = saida_classificada[saida_classificada.index.isin(dados.index)]

    classificações = pd.DataFrame(index=range(len(individuo)), columns=[(*classes), "CF", "classe"])
    
    for k in classes:
        indices = np.where(saida_classificada==k)[0]
        classificações.loc[:, k] = np.sum(uj.iloc[indices, :])

    indices = np.where(np.max(classificações.loc[:, classes], axis=1)==0)[0]
    classificações.loc[indices,"classe"] = "Indeterminado"
    classificações.loc[indices,"CF"] = 0
    indices = np.where(np.max(classificações.loc[:, classes], axis=1)!=0)[0]
    classificações.loc[indices,"classe"] = pd.DataFrame.idxmax(classificações.loc[indices, classes],axis=1)
    
    for k in classes:
        indices = np.where(classificações.loc[:,"classe"]==k)[0]
        beta_barra = np.sum(classificações.loc[indices, classes[classes != k]], axis=1) / (len(classes) -1)
        classificações.loc[indices,"CF"] = (classificações.loc[indices, k] - beta_barra) \
                                            / np.sum(classificações.loc[indices, classes], axis=1)
          
    return classificações

def estima_saida(dados, uj, classificacao, saida_classificada=pd.DataFrame()):
    """Função que estima a saida para um conjunto de regras. Recebe como entradas o dataframe de dados,
    as pertinências às regras uj, a classificação do treinamento para 1 indivíduo e pode ou não receber a saída
    real classificada. 
    Caso não receba como entrada a saída real classificada, irá retornar um vetor com a classificação da saída estimada
    para cada ponto.
    Caso receba a saída classificada real, irá retornar um vetor contendo o número de erros e o percentual de erros
    na classificação estimada.
    """    
    aux = uj*classificacao.loc[:,"CF"]

    saida_estimada = pd.DataFrame(index=dados.index, columns=["estimado"])
    saida_estimada.loc[:, "estimado"] = list(classificacao.loc[pd.DataFrame.idxmax(aux,axis=1), "classe"])

    indices = np.where(aux.sum(axis=1)==0)[0]
    saida_estimada.iloc[indices,:] = "manter"

    if not(saida_classificada.empty):
        saida_estimada["real"] = saida_classificada    
        erros = len(np.where(saida_estimada.loc[:,"estimado"]!=saida_estimada.loc[:,"real"])[0])
        percentual_erros = erros / len(saida_estimada)
        return [erros, percentual_erros]
    else:
        return saida_estimada