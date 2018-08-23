# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 00:36:07 2017

@author: rafaaeraf
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:52:13 2017

@author: rafaaeraf
"""

#import nebulosos as neb
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

tempo = time.time()
## Inicializando as classes
#creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#creator.create("Individual", list, fitness=creator.FitnessMax)

def calcula_lucro(individuo, matriz_pertinencias, dados, saida_classificada, classes, taxa, data, regras, regras_classe, regras_CF):
    uj = nebulosos.calcula_uj(matriz_pertinencias, individuo)
    
    classificacao = pd.DataFrame(index=range(len(individuo)), columns=[(*classes), "CF", "classe"])
    for i in range(len(individuo)):
        classificacao.loc[i,"CF"] = regras_CF[regras.index(individuo[i])]
        classificacao.loc[i,"classe"] = regras_classe[regras.index(individuo[i])]
    
    saida = nebulosos.estima_saida(dados, uj, classificacao)

    saldo = pd.DataFrame(index=dados.index, columns=["saida","variacao","holdando_moeda", \
                                                        "saldo", "saldo_holdar","cotacao_BTC"])
    saldo.loc[saldo.index[0],"saldo"] = 1 #saldo em BTC
    saldo.loc[saldo.index[0],"holdando_moeda"] = "sim" #saldo em BTC
    saldo.loc[saldo.index[0],"saldo_holdar"] = 1 #saldo em BTC para sempre holdar a moeda
    saldo.loc[:,"saida"] = saida
    saldo.loc[:,"variacao"] = taxa.loc[dados.index,(dados.columns[-1]).replace("saida_","")]
    saldo.loc[:,"cotacao_BTC"] = data.loc[dados.index,"XXBTZUSD"]
    taxa_maker = 0.16/100
    
    for i in range(len(saldo)-1):
        if saldo.loc[saldo.index[i],"saida"]=="manter":
            if saldo.loc[saldo.index[i],"holdando_moeda"]=="sim":
                saldo.loc[saldo.index[i+1],"saldo"] = saldo.loc[saldo.index[i],"saldo"]*(1+saldo.loc[saldo.index[i+1],"variacao"])
                saldo.loc[saldo.index[i+1],"holdando_moeda"] = "sim"
            elif saldo.loc[saldo.index[i],"holdando_moeda"]=="não":
                saldo.loc[saldo.index[i+1],"saldo"] = saldo.loc[saldo.index[i],"saldo"]
                saldo.loc[saldo.index[i+1],"holdando_moeda"] = "não"

        elif saldo.loc[saldo.index[i],"saida"]=="vender":
            if saldo.loc[saldo.index[i],"holdando_moeda"]=="sim":
                #taxa de venda
                saldo.loc[saldo.index[i+1],"saldo"] = saldo.loc[saldo.index[i],"saldo"]*(1-taxa_maker)
                saldo.loc[saldo.index[i+1],"holdando_moeda"] = "não"
            elif saldo.loc[saldo.index[i],"holdando_moeda"]=="não":
                saldo.loc[saldo.index[i+1],"saldo"] = saldo.loc[saldo.index[i],"saldo"]
                saldo.loc[saldo.index[i+1],"holdando_moeda"] = "não"

        elif saldo.loc[saldo.index[i],"saida"]=="comprar":
            if saldo.loc[saldo.index[i],"holdando_moeda"]=="sim":
                saldo.loc[saldo.index[i+1],"saldo"] = saldo.loc[saldo.index[i],"saldo"]*(1+saldo.loc[saldo.index[i+1],"variacao"])
                saldo.loc[saldo.index[i+1],"holdando_moeda"] = "sim"
            elif saldo.loc[saldo.index[i],"holdando_moeda"]=="não":
                #taxa de compra
                saldo.loc[saldo.index[i+1],"saldo"] = saldo.loc[saldo.index[i],"saldo"]*(1+saldo.loc[saldo.index[i+1],"variacao"]) \
                                                                                        *(1-taxa_maker)
                saldo.loc[saldo.index[i+1],"holdando_moeda"] = "sim"

        saldo.loc[saldo.index[i+1],"saldo_holdar"] = saldo.loc[saldo.index[i],"saldo_holdar"]* \
                                                                                    (1+saldo.loc[saldo.index[i+1],"variacao"])

        saldo.loc[:,"saldo_dolar"] = saldo.loc[:,"saldo"]*saldo.loc[:,"cotacao_BTC"]
        saldo.loc[:,"saldo_holdar_dolar"] = saldo.loc[:,"saldo_holdar"]*saldo.loc[:,"cotacao_BTC"]    

    return saldo


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


lucro_best_ind_all = {}
lucro_hof_all = {}

for saida in saidas_usadas:
    lucro_best_ind = []
    lucro_hof = []    
    for i in range(len(hof_all[saida])):
        saldo = calcula_lucro(hof_all[saida][i], matriz_pertinencias_new[saida], dados_new[saida], saida_classificada_new[saida], 
                              classes_new, taxa_new, data_new, regras[saida], regras_classe[saida], regras_CF[saida])
        lucro_hof.append(saldo)
        
    for i in range(len(best_ind_all[saida])):
        saldo = calcula_lucro(best_ind_all[saida][i], matriz_pertinencias_new[saida], dados_new[saida], saida_classificada_new[saida], 
                              classes_new, taxa_new, data_new, regras[saida], regras_classe[saida], regras_CF[saida])
        lucro_best_ind.append(saldo)
    
    lucro_hof_all[saida] = lucro_hof
    lucro_best_ind_all[saida] = lucro_best_ind
    print(saida)
        
lucro['lucro_best_ind_all'] = lucro_best_ind_all
lucro['lucro_hof_all'] = lucro_hof_all

#lucro_best_ind_all = lucro['lucro_best_ind_all']
#lucro_hof_all = lucro['lucro_hof_all']

saldo_final_all = {}
saldo_final_holdar_all = {}
tabela_saldo_final_hof = pd.DataFrame(index=saidas_usadas, columns=["Max Fit Validação (%)", "Saldo Final (BTC)", "Saldo Final Holdar (BTC)", "Relação Saldo Final/Saldo Final Holdar (%)"])
#tabela_saldo_final_hof = pd.DataFrame(index=saidas_usadas, columns=["Saldo Final (BTC)", "Saldo Final Holdar (BTC)", "Saldo Final (USD)",
#                                                                    "Saldo Final Holdar (USD)", "Relação Saldo Final/Saldo Final Holdar (%)", "Max Fit Validação (%)"])

# tabela hof
for saida in lucro_hof_all:
    tabela_saldo_final_hof.loc[saida,"Saldo Final (BTC)"] = lucro_hof_all[saida][0].iloc[-1, 3]
    tabela_saldo_final_hof.loc[saida,"Saldo Final Holdar (BTC)"] = lucro_hof_all[saida][0].iloc[-1, 4]
#    tabela_saldo_final_hof.loc[saida,"Saldo Final (USD)"] = lucro_hof_all[saida][0].iloc[-1, 6]
#    tabela_saldo_final_hof.loc[saida,"Saldo Final Holdar (USD)"] = lucro_hof_all[saida][0].iloc[-1, 7]
    tabela_saldo_final_hof.loc[saida,"Relação Saldo Final/Saldo Final Holdar (%)"] = lucro_hof_all[saida][0].iloc[-1, 6] / lucro_hof_all[saida][0].iloc[-1, 7]
    tabela_saldo_final_hof.loc[saida,"Max Fit Validação (%)"] = max(max_fits_all.loc[:, saida])
    tabela_saldo_final_hof = tabela_saldo_final_hof.sort_values(by=["Max Fit Validação (%)"], ascending=False)
    
# evolução do algorítmo genético
for saida in lucro_best_ind_all:
    saldo_final = []
    saldo_final_holdar = []
    for i in range(len(lucro_best_ind_all[saida])):
        saldo_final.append(lucro_best_ind_all[saida][i].iloc[-1, 3])
        saldo_final_holdar.append(lucro_best_ind_all[saida][i].iloc[-1, 4])
        
    saldo_final_all[saida] = saldo_final
    saldo_final_holdar_all[saida] = saldo_final_holdar

figura = plt.figure()
ax = plt.subplot(111)
for saida in saldo_final_all:
    plt.plot(range(len(saldo_final_all[saida])), 100*np.array(saldo_final_all[saida]) / np.array(saldo_final_holdar_all[saida]))

 # Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(saldo_final_all.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

plt.title("Evolução do Lucro por Geração - De 11/11/17 a 03/12/17")
plt.xlabel("Geração")
plt.ylabel("Relação Saldo Final/Saldo Final Holdar (%)")
plt.show()

figura.savefig("D:/Dropbox/Nebuloso/Artigo/Artigo/Imagens/semfee/lucroxgeracao.eps") #mudar para eps


# valor no tempo melhor resultado
melhor = np.argmax(tabela_saldo_final_hof.loc[:,"Relação Saldo Final/Saldo Final Holdar (%)"])
figura = plt.figure()
ax = plt.subplot(111)
plt.plot(lucro_hof_all[melhor][0].index, lucro_hof_all[melhor][0].iloc[:,3])
plt.plot(lucro_hof_all[melhor][0].index, lucro_hof_all[melhor][0].iloc[:,4])

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# Put a legend to the right of the current axis
ax.legend(["Saldo (BTC)", "Saldo Holdar (BTC)"], loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=5)

plt.title("Saldo Ao Longo do Tempo - %s" %melhor)
plt.xlabel("Data")
plt.ylabel("Saldo (BTC)")

loc = plticker.MultipleLocator(base=4.0) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%d/%y'))

plt.show()

figura.savefig("D:/Dropbox/Nebuloso/Artigo/Artigo/Imagens/semfee/saldoxtempo.eps") #mudar para eps

tempo = time.time() - tempo
print(tempo)

