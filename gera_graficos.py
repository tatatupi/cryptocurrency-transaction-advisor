# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:14:50 2017

@author: rafaaeraf
"""

import genetico
import shelve 
import numpy as np
import time
import random
import nebulosos
import copy
import pandas as pd
from deap import base, creator, tools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as plticker
import matplotlib.dates as mdates

# Inicializando as classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_ind", genetico.attr_individuo, regras)
toolbox.register("individual", genetico.initPop, creator.Individual, n_regras_par, regras_classe, regras)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", genetico.calc_acerto)
toolbox.register("evaluate", genetico.calc_acerto)
toolbox.register("mate", genetico.cruzamento)
toolbox.register("mutate", genetico.mutacao)
toolbox.register("select", tools.selTournament, tournsize=3)

database = shelve.open('all_data') 
database_new = shelve.open('all_data_new')
resultados = shelve.open('results_100-50',flag='r')
lucro = shelve.open('lucro')
#resultados2 = shelve.open('D:/Dropbox/Nebuloso/Trabalho final/code/RESULTADOS_5-12_pop100_ger40/results')

data_new = database_new['data']
taxa_new = database_new['taxa']
entrada_new = database_new['entrada']
variaveis_linguisticas_new = database_new['variaveis_linguisticas']
classes_new = database_new['classes']
dados_new = database_new['dados']
matriz_pertinencias_new = database_new['matriz_pertinencias']
saida_classificada_new = database_new['saida_classificada']

max_fits_all = resultados['max_fits']
min_fits_all = resultados['min_fits']
mean_fits_all = resultados['mean_fits']
std_fits_all = resultados['std_fits']
best_ind_all = resultados['best_ind']
hof_all = resultados['hof']

regras = database['regras']
regras_classe = database['regras_classe']
regras_CF = database['regras_CF']

saidas_usadas = copy.deepcopy(entrada_new)
saidas_usadas.remove("XXBTZUSD")

lucro_best_ind_all = lucro['lucro_best_ind_all']
lucro_hof_all = lucro['lucro_hof_all']

saldo_final_all = {}
saldo_final_holdar_all = {}
tabela_saldo_final_hof = pd.DataFrame(index=saidas_usadas, columns=["Max Fit Validação (%)", "Saldo Final Algoritmo (BTC)", "Saldo Final Manter (BTC)", "Relação Saldo Final Algoritmo/Saldo Final Manter (%)"])
#tabela_saldo_final_hof = pd.DataFrame(index=saidas_usadas, columns=["Saldo Final (BTC)", "Saldo Final Holdar (BTC)", "Saldo Final (USD)",
#                                                                    "Saldo Final Holdar (USD)", "Relação Saldo Final/Saldo Final Holdar (%)", "Max Fit Validação (%)"])

# tabela hof
for saida in lucro_hof_all:
    tabela_saldo_final_hof.loc[saida,"Saldo Final Algoritmo (BTC)"] = lucro_hof_all[saida][0].iloc[-1, 3]
    tabela_saldo_final_hof.loc[saida,"Saldo Final Manter (BTC)"] = lucro_hof_all[saida][0].iloc[-1, 4]
#    tabela_saldo_final_hof.loc[saida,"Saldo Final (USD)"] = lucro_hof_all[saida][0].iloc[-1, 6]
#    tabela_saldo_final_hof.loc[saida,"Saldo Final Holdar (USD)"] = lucro_hof_all[saida][0].iloc[-1, 7]
    tabela_saldo_final_hof.loc[saida,"Relação Saldo Final Algoritmo/Saldo Final Manter (%)"] = lucro_hof_all[saida][0].iloc[-1, 6] / lucro_hof_all[saida][0].iloc[-1, 7]
    tabela_saldo_final_hof.loc[saida,"Max Fit Validação (%)"] = max(max_fits_all.loc[:, saida])
    tabela_saldo_final_hof = tabela_saldo_final_hof.sort_values(by=["Max Fit Validação (%)"], ascending=False)
    
## evolução do algorítmo genético 1
#moedas1 = np.array(tabela_saldo_final_hof.index)
#moedas2 = list(moedas1[7:])
#moedas1 = list(moedas1[0:5])
##moedas1 = [tabela_saldo_final_hof.index[i] for i in range(8)]
##moedas2 = [tabela_saldo_final_hof.index[i] for i in range(8,len(tabela_saldo_final_hof.index))]
#
#for saida in moedas1:
#    saldo_final = []
#    saldo_final_holdar = []
#    for i in range(len(lucro_best_ind_all[saida])):
#        saldo_final.append(lucro_best_ind_all[saida][i].iloc[-1, 3])
#        saldo_final_holdar.append(lucro_best_ind_all[saida][i].iloc[-1, 4])
#        
#    saldo_final_all[saida] = saldo_final
#    saldo_final_holdar_all[saida] = saldo_final_holdar
#
#figura = plt.figure()
#ax = plt.subplot(111)
#for saida in moedas1:
#    plt.plot(range(len(saldo_final_all[saida])), 100*np.array(saldo_final_all[saida]) / np.array(saldo_final_holdar_all[saida]))
#
# # Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
## Put a legend to the right of the current axis
#ax.legend(moedas1, loc='center left', bbox_to_anchor=(1, 0.5))
#
##plt.title("Evolução do Lucro por Geração (Melhores Fitness) - 11/11/17 a 03/12/17")
#plt.xlabel("Geração")
#plt.ylabel("Relação Saldo Final Algoritmo/\nSaldo Final Manter (%)",)
#plt.show()
#
#figura.savefig("D:/Dropbox/Nebuloso/Artigo/Artigo/Imagens/lucroxgeracao5moedas.eps") #mudar para eps
#
## evolução do algorítmo genético 2
#for saida in moedas2:
#    saldo_final = []
#    saldo_final_holdar = []
#    for i in range(len(lucro_best_ind_all[saida])):
#        saldo_final.append(lucro_best_ind_all[saida][i].iloc[-1, 3])
#        saldo_final_holdar.append(lucro_best_ind_all[saida][i].iloc[-1, 4])
#        
#    saldo_final_all[saida] = saldo_final
#    saldo_final_holdar_all[saida] = saldo_final_holdar
#
#figura = plt.figure()
#ax = plt.subplot(111)
#for saida in moedas2:
#    plt.plot(range(len(saldo_final_all[saida])), 100*np.array(saldo_final_all[saida]) / np.array(saldo_final_holdar_all[saida]))
#
# # Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
## Put a legend to the right of the current axis
#ax.legend(moedas2, loc='center left', bbox_to_anchor=(1, 0.5))
#
##plt.title("Evolução do Lucro por Geração (Piores Fitness) - 11/11/17 a 03/12/17")
#plt.xlabel("Geração")
#plt.ylabel("Relação Saldo Final Algoritmo/\nSaldo Final Manter (%)")
#plt.show()
#
##figura.savefig("D:/Dropbox/Nebuloso/Artigo/Artigo/Imagens/lucroxgeracao2.eps") #mudar para eps
#




# evolução do algorítmo genético 1
moedas1 = np.array(tabela_saldo_final_hof.index)
moedas2 = list(moedas1[-5:])
moedas1 = list(moedas1[0:5])
#moedas1 = [tabela_saldo_final_hof.index[i] for i in range(8)]
#moedas2 = [tabela_saldo_final_hof.index[i] for i in range(8,len(tabela_saldo_final_hof.index))]


# # Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
## Put a legend to the right of the current axis
#ax.legend(moedas1, loc='center left', bbox_to_anchor=(1, 0.5))

#MEDIO MELHORES
figura = plt.figure()
ax = plt.subplot(111)
for saida in moedas1:
    plt.plot(range(len(mean_fits_all[saida])), mean_fits_all.loc[:,saida])



 # Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])

# Put a legend to the right of the current axis
ax.legend(moedas1, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)


#plt.title("Evolução do Lucro por Geração (Melhores Fitness) - 11/11/17 a 03/12/17")
plt.xlabel("Generation")
plt.ylabel("Validation Average Fitness",)
plt.show()

figura.savefig("D:/Dropbox/Nebuloso/Artigo/Artigo/Imagens/fitnessxgeracao_media_melhor.eps") #mudar para eps



#MAX MELHORES
figura = plt.figure()
ax = plt.subplot(111)
for saida in moedas1:
    plt.plot(range(len(max_fits_all[saida])), max_fits_all.loc[:,saida])



 # Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])

# Put a legend to the right of the current axis
ax.legend(moedas1, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)


#plt.title("Evolução do Lucro por Geração (Melhores Fitness) - 11/11/17 a 03/12/17")
plt.xlabel("Generation")
plt.ylabel("Validation Maximum Fitness",)
plt.show()

figura.savefig("D:/Dropbox/Nebuloso/Artigo/Artigo/Imagens/fitnessxgeracao_max_melhor.eps") #mudar para eps



#MEDIO PIORES
figura = plt.figure()
ax = plt.subplot(111)
for saida in moedas2:
    plt.plot(range(len(mean_fits_all[saida])), mean_fits_all.loc[:,saida])



 # Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])

# Put a legend to the right of the current axis
ax.legend(moedas2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)


#plt.title("Evolução do Lucro por Geração (Melhores Fitness) - 11/11/17 a 03/12/17")
plt.xlabel("Generation")
plt.ylabel("Validation Average Fitness",)
plt.show()

figura.savefig("D:/Dropbox/Nebuloso/Artigo/Artigo/Imagens/fitnessxgeracao_media_pior.eps") #mudar para eps



#MAX PIORES
figura = plt.figure()
ax = plt.subplot(111)
for saida in moedas2:
    plt.plot(range(len(max_fits_all[saida])), max_fits_all.loc[:,saida])



 # Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])

# Put a legend to the right of the current axis
ax.legend(moedas2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)


#plt.title("Evolução do Lucro por Geração (Melhores Fitness) - 11/11/17 a 03/12/17")
plt.xlabel("Generation")
plt.ylabel("Validation Maximum Fitness",)
plt.show()

figura.savefig("D:/Dropbox/Nebuloso/Artigo/Artigo/Imagens/fitnessxgeracao_max_pior.eps") #mudar para eps




# valor no tempo melhor resultado
melhor = np.argmax(tabela_saldo_final_hof.loc[:,"Max Fit Validação (%)"])
#melhor = np.argmax(tabela_saldo_final_hof.loc[:,"Max Fit Validação (%)"])
figura = plt.figure()
ax = plt.subplot(111)
plt.plot(lucro_hof_all[melhor][0].index, lucro_hof_all[melhor][0].iloc[:,3])
plt.plot(lucro_hof_all[melhor][0].index, lucro_hof_all[melhor][0].iloc[:,4])

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# Put a legend to the right of the current axis
ax.legend(["Trading Balance (BTC)", "Holding Balance (BTC)"], loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=5)

#plt.title("Saldo Ao Longo do Tempo - %s" %melhor)
plt.xlabel("Date")
plt.ylabel("Balance (BTC)")

loc = plticker.MultipleLocator(base=3.5) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))

plt.show()

figura.savefig("D:/Dropbox/Nebuloso/Artigo/Artigo/Imagens/saldoxtempo.eps") #mudar para eps




# valor no tempo melhor resultado
melhor_saldo = np.argmax(tabela_saldo_final_hof.loc[:,"Relação Saldo Final Algoritmo/Saldo Final Manter (%)"])
figura = plt.figure()
ax = plt.subplot(111)
plt.plot(lucro_hof_all[melhor_saldo][0].index, lucro_hof_all[melhor_saldo][0].iloc[:,3])
plt.plot(lucro_hof_all[melhor_saldo][0].index, lucro_hof_all[melhor_saldo][0].iloc[:,4])

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# Put a legend to the right of the current axis
ax.legend(["Trading Balance (BTC)", "Holding Balance (BTC)"], loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=5)

#plt.title("Saldo Ao Longo do Tempo - %s" %melhor)
plt.xlabel("Date")
plt.ylabel("Balance (BTC)")

loc = plticker.MultipleLocator(base=3.5) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))

plt.show()

figura.savefig("D:/Dropbox/Nebuloso/Artigo/Artigo/Imagens/saldoxtempoGNO.eps") #mudar para eps





# janela de decisão
holdando = lucro_hof_all[melhor][0].loc[:,"holdando_moeda"].copy(deep=True)
saldo_algoritmo = lucro_hof_all[melhor][0].loc[:,"saldo"].copy(deep=True)
saldo_holdar = lucro_hof_all[melhor][0].loc[:,"saldo_holdar"].copy(deep=True)

cortemin = 100
cortemax = round(cortemin + 2*24*60/15)
holdando = holdando.iloc[cortemin:cortemax]
saldo_algoritmo = saldo_algoritmo.iloc[cortemin:cortemax]
saldo_holdar = saldo_holdar.iloc[cortemin:cortemax]

maximo = np.max([saldo_algoritmo, saldo_holdar])
minimo = np.min([saldo_algoritmo, saldo_holdar])
aux = np.where(holdando=="sim")[0]
holdando[aux] = maximo
aux = np.where(holdando=="não")[0]
holdando[aux] = minimo

figura = plt.figure()
ax = plt.subplot(111)
plt.plot(holdando.index, saldo_algoritmo)
plt.plot(holdando.index, saldo_holdar)
plt.plot(holdando.index, holdando)

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# Put a legend to the right of the current axis
ax.legend(["Trading Balance (BTC)", "Holding Balance (BTC)", "Buy / Sell"], loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=3)

#plt.title("Saldo Ao Longo do Tempo - %s" %melhor)
plt.xlabel("Date")
plt.ylabel("Balance (BTC)")

loc = plticker.MultipleLocator(base=1) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))

plt.show()

figura.savefig("D:/Dropbox/Nebuloso/Artigo/Artigo/Imagens/detalhedecisoes.eps") #mudar para eps



estatisticas = pd.DataFrame(index = moedas1, columns = ["Comprar (%)", "Manter (%)", "Vender (%)", "Tempo com Moeda (%)", "Tempo sem Moeda (%)"])
for saida in moedas1:
    estatisticas.loc[saida, "Comprar (%)"] = len(np.where(lucro_hof_all[saida][0].loc[:,"saida"]=="comprar")[0]) / len(lucro_hof_all[saida][0])
    estatisticas.loc[saida, "Manter (%)"] = len(np.where(lucro_hof_all[saida][0].loc[:,"saida"]=="manter")[0]) / len(lucro_hof_all[saida][0])
    estatisticas.loc[saida, "Vender (%)"] = len(np.where(lucro_hof_all[saida][0].loc[:,"saida"]=="vender")[0]) / len(lucro_hof_all[saida][0])
    estatisticas.loc[saida, "Tempo com Moeda (%)"] = len(np.where(lucro_hof_all[saida][0].loc[:,"holdando_moeda"]=="sim")[0]) / len(lucro_hof_all[saida][0])
    estatisticas.loc[saida, "Tempo sem Moeda (%)"] = len(np.where(lucro_hof_all[saida][0].loc[:,"holdando_moeda"]=="não")[0]) / len(lucro_hof_all[saida][0])
    
    estatisticas.loc[saida, "Comprar"] = len(np.where(lucro_hof_all[saida][0].loc[:,"saida"]=="comprar")[0])
    estatisticas.loc[saida, "Vender"] = len(np.where(lucro_hof_all[saida][0].loc[:,"saida"]=="vender")[0])
    estatisticas.loc[saida, "Manter"] = len(np.where(lucro_hof_all[saida][0].loc[:,"saida"]=="manter")[0])