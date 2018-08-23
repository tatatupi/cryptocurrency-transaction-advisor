# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:33:40 2017

@author: Taiguara
"""
import nebulosos as neb

def gera_regras(entrada,variaveis_linguisticas,matriz_pertinencias,dados_trein,saida_classificada,classes):    
    antecedentes = []
    for i in entrada:
        for j in variaveis_linguisticas:
            aux = []
            aux.append(i)
            aux.append(j)
            
            antecedentes.append(aux)
    
    ## dois antecedentes
    atributos = []
    for ant1 in antecedentes:
        for ant2 in antecedentes:
            if ant1[0] != ant2[0]:
                atr = []
                aux = []
                aux.append(ant1[0])
                aux.append(ant1[1])
                aux.append(ant2[0])
                aux.append(ant2[1])
                
                atr.append(aux)
                
                atributos.append(atr)
    
    ## um antecedente
    for ant in antecedentes:
        atr=[]
        atr.append(aux)
        atributos.append(atr)
    
    classificacoes = []
    
    regras=[]
    regras_classe=[]
    regras_CF=[]
    
    for individuo in atributos:
        uj = neb.calcula_uj(matriz_pertinencias, individuo)
        aux = neb.realiza_treinamento(dados_trein, saida_classificada, uj, individuo, classes)
        if aux['classe'][0]!="manter":
            regras.append(individuo[0])
            regras_classe.append(aux['classe'][0])
            regras_CF.append(aux['CF'][0])
            
    return regras, regras_classe, regras_CF



