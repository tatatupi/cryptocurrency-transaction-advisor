# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:31:10 2017

@author: Taiguara
"""
import nebulosos as neb
import pandas as pd
import numpy as np
import random
import copy
import deap 

def remove_duplicates(item):
    outlist=[]

    for element in item:     
      if element not in outlist:
        outlist.append(element)

    return outlist

def initPop(container, n_regras_par, regras_classe, regras):
    n = random.choice(n_regras_par)
    individuo=[]
    for i in range(n-1):
        reg=random.choice(regras)
        while reg in individuo:
            reg=random.choice(regras)
        individuo.append(reg)
        
    classes=[]
    for ant in individuo:
        classes.append(regras_classe[regras.index(ant)])
        
    aux=0
    if "comprar" not in classes:
        while aux != "comprar":
            i = np.random.randint(0, len(regras))
            aux = regras_classe[i]   
    elif "vender" not in classes:
        while aux != "vender":
            i = np.random.randint(0, len(regras))
            aux = regras_classe[i]   
    else:
        i = np.random.randint(0, len(regras))
    
    individuo.append(regras[i])
    
    return container(individuo)


def attr_individuo(regras):
    
    atributo = regras[random.randint(0,len(regras))]

    return atributo;

def calc_acerto(individuo, dados_valid, saida_classificada, classes, matriz_pertinencias, regras_classe, regras_CF, regras):
    """Função que estima a saida para um conjunto de regras. Recebe como entradas o dataframe de dados,
    as pertinências às regras uj, a classificação do treinamento para 1 indivíduo e pode ou não receber a saída
    real classificada. 
    Caso não receba como entrada a saída real classificada, irá retornar um vetor com a classificação da saída estimada
    para cada ponto.
    Caso receba a saída classificada real, irá retornar um vetor contendo o número de erros e o percentual de erros
    na classificação estimada.
    """   
    uj = neb.calcula_uj(matriz_pertinencias, individuo)    
    classificacao=pd.DataFrame()
    i=0
    for ant in individuo:
        classificacao.loc[i,'CF']=(regras_CF[regras.index(ant)])
        classificacao.loc[i,'classe']=(regras_classe[regras.index(ant)])
        i+=1
      
    uj = uj[uj.index.isin(dados_valid.index)]
    aux = uj*classificacao.loc[:,"CF"]
    
    saida_estimada = pd.DataFrame(index=dados_valid.index, columns=["estimado"])
    saida_estimada.loc[:, "estimado"] = list(classificacao.loc[pd.DataFrame.idxmax(aux,axis=1), "classe"])

    indices = np.where(aux.sum(axis=1)==0)[0]
    saida_estimada.iloc[indices,:] = "manter"

    saida_estimada["real"] = saida_classificada    
    erros = len(np.where(saida_estimada.loc[:,"estimado"]!=saida_estimada.loc[:,"real"])[0])
    percentual_erros = erros / len(saida_estimada)
    acerto = 1-percentual_erros
    
    return acerto,

def cruzamento(pai1, pai2, regras_classe, regras):
    if len(pai1)<3:
        pai1.append(regras[random.randint(0,len(regras))])
    if len(pai2)<3:
        pai2.append(regras[random.randint(0,len(regras))])

    n_regras_1 = np.random.choice([len(remove_duplicates(pai1)),len(remove_duplicates(pai2))])
    n_regras_2 = np.random.choice([len(remove_duplicates(pai1)),len(remove_duplicates(pai2))])

    aux = pai1+pai2
    vetor_regras = []
    #evita regras repetidas
    for i in aux:
           if i not in vetor_regras:
                vetor_regras.append(i)
    
    #seleciona uma regra comprar
    aux = 0
    while aux != "comprar":
        i = np.random.randint(0, len(vetor_regras))
        aux = regras_classe[regras.index(vetor_regras[i])]   
    aux1 = [i]       
    #seleciona uma regra vender
    aux = 0
    while aux != "vender":
        i = np.random.randint(0, len(vetor_regras))
        aux = regras_classe[regras.index(vetor_regras[i])]   
    aux1.append(i)
    
    restantes = list(range(len(vetor_regras)))
    restantes.remove(aux1[0])
    restantes.remove(aux1[1])

    aux1.extend(list(np.random.choice(restantes, size=n_regras_1-2, replace=False)))        
    
    filho1 = [vetor_regras[i] for i in aux1]
    
    #seleciona uma regra comprar
    aux = 0
    while aux != "comprar":
        i = np.random.randint(0, len(vetor_regras))
        aux = regras_classe[regras.index(vetor_regras[i])]   
    aux2 = [i]       
    #seleciona uma regra vender
    aux = 0
    while aux != "vender":
        i = np.random.randint(0, len(vetor_regras))
        aux = regras_classe[regras.index(vetor_regras[i])]   
    aux2.append(i)
    
    restantes = list(range(len(vetor_regras)))
    restantes.remove(aux2[0])
    restantes.remove(aux2[1])
    
    aux2.extend(list(np.random.choice(restantes, size=n_regras_2-2, replace=False)))
    filho2 = [vetor_regras[i] for i in aux2]
    
    return [filho1, filho2]

def mutacao(filho, regras_classe, regras):
    
    if len(filho)<3:
        filho.append(regras[random.randint(0,len(regras))])
    
    aux = copy.deepcopy(filho)
    regra_mutada = np.random.randint(0,len(aux))
    
    demais_classes = []
    for i in range(len(aux)):
        demais_classes.append(regras_classe[regras.index(filho[i])])
    classe_mutada = demais_classes[regra_mutada]
    del demais_classes[regra_mutada]
    
    if classe_mutada in demais_classes:
        aux1 = np.random.randint(0,len(regras))
        aux[regra_mutada] = regras[aux1]
    else:
        aux2 = 0
        while aux2 != classe_mutada:
            aux1 = np.random.randint(0, len(regras))
            aux2 = regras_classe[aux1]
        aux[regra_mutada] = regras[aux1]
    
    return aux