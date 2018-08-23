import pandas as pd
import numpy as np
from pandas import MultiIndex
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import time
from datetime import datetime, timedelta
import random

def unixtime2str(unixtime):
    strtime = datetime.fromtimestamp(int(unixtime))
    return strtime

def taxa_dados(data,periodo, periodo_saida):

    date_index=[]
    for i in range(data.iloc[:,0].size):
        date_index.append(unixtime2str(data.iloc[i,0]))

    data = data.drop(['Unnamed: 0'],axis=1)
    data = data.set_index([date_index])
    
    # reamostragem com o último valor
    data_reamostrada = data.resample(str(periodo)+'T').last()
    
    datadif = data_reamostrada.shift(-1, freq = str(periodo)+'T').subtract(data_reamostrada) # shift na série temporal, para subtração
    datadif = datadif.shift(1, freq = str(periodo)+'T') # volta a série temporal para o índice correto

    taxa = datadif.divide(data_reamostrada.shift(1,freq=str(periodo)+'T'))  # divide as diferenças pelo valor do índice anterior
    
    return taxa

def calc_saida(taxa,periodo,periodo_saida):
    correlacao = []
    corr_max = []
    taxa_aux = []
    matriz_corr =  pd.DataFrame(columns=taxa.columns,index=taxa.columns)
    
    for name in list(taxa.columns.values):
        taxa_aux = taxa.copy(deep=True)
        moeda =  taxa_aux[name].shift(1,freq = str(periodo)+'T') #shift na moeda
        taxa_aux[name+str('Shift')] = moeda #inclui a moeda com o shift
        
        correlacao = taxa_aux.corr().replace(1,0).abs() #calcula a correlacao e remove as diagonais (sempre 1)
        matriz_corr.loc[name,:] = correlacao.iloc[0:-1,-1] #armazena todas as correlações
        correlacao = correlacao[name+str('Shift')].sort_values(ascending=False) #ordena em ordem decrescente
        
        linha = []
        linha.append(name)
        for i in range(len(list(taxa.columns))):
            linha.append(correlacao[i])
      
        corr_max.append(linha)
    
    corr_max = pd.DataFrame(corr_max).set_index(0).mean(axis=1).sort_values(ascending=False) #ordena as moedas por maior média de correlação

    saida = corr_max.index[0] #a moeda escolhida como saída é a de maior correlação
    
    return saida

def prep_dados(saida,entrada,taxa,periodo):
    # Dados de Entrada e Saída
    Y = pd.DataFrame(taxa[saida].shift(1,freq = str(periodo)+'T'))

    X = pd.DataFrame(index=taxa.index)
    for i in range(len(entrada)):
        X.loc[:,entrada[i]]=pd.DataFrame(taxa[entrada[i]])

    Y.columns=["saida_" + Y.columns[0]]
    dados = X.join(Y).dropna()

#    dados_norm = (dados - dados.mean()) / (dados.max() - dados.min())
    
    return dados






