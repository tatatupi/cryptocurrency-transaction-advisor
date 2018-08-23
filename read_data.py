import nebulosos as neb
import prep_dados as dd
import gera_regras as gr
import numpy as np
import time
import pandas as pd
import shelve

tempo = time.time()

database = shelve.open('all_data') 

trein_ratio = 0.7
valid_ratio = 1-trein_ratio

periodo = 15
periodo_saida = 15

classes = np.array(["vender", "manter", "comprar"])
cai_muito = -1.5 # de -infinito a...
cai_pouco = -0.5 # de cai muito a...
estavel = 0.5 # de cai pouco a...
sobe_pouco = 1.5 # de estável a...
# sobe_muito = de sobe_pouco a + infinito
vender = -0.2 # de -infinito a...
manter = 0.2  # de vender a...
# comprar = de manter a + infinito

limites_pertinencia = [cai_muito, cai_pouco, estavel, sobe_pouco]
parametros_pertinencia = neb.calc_par_pertinencia(limites_pertinencia)
variaveis_linguisticas = parametros_pertinencia.index.values

file = "dados_final1.csv"
data = pd.read_csv(file, sep=';',decimal=',')
taxa = dd.taxa_dados(data,periodo,periodo_saida)
entrada = list(taxa.columns)

database['data'] = data
database['taxa'] = taxa
database['entrada'] = entrada
database['variaveis_linguisticas'] = variaveis_linguisticas
database['classes'] = classes

dados = {}
matriz_pertinencias = {}
saida_classificada = {}
dados_trein = {}
dados_valid = {}
regras = {}
regras_classe = {}
regras_CF = {}

for saida in entrada:
    print("Começando saída " + saida)
    dados[saida] = dd.prep_dados(saida,entrada,taxa,periodo)
    matriz_pertinencias[saida] = neb.calc_pertinencia(dados[saida],parametros_pertinencia)
    saida_classificada[saida] = neb.classifica_saida(dados[saida],vender,manter,classes)
    dados_aux = dados[saida].sample(frac=1)
    dados_trein[saida] = dados_aux[0:int(np.round(len(dados[saida])*trein_ratio))-1]
    dados_valid[saida] = dados_aux[int(np.round(len(dados[saida])*trein_ratio)):len(dados[saida])]
    regras[saida],regras_classe[saida],regras_CF[saida] = gr.gera_regras(entrada,variaveis_linguisticas,matriz_pertinencias[saida],dados_trein[saida],saida_classificada[saida],classes)

database['dados'] = dados
database['matriz_pertinencias'] = matriz_pertinencias
database['saida_classificada'] = saida_classificada
database['dados_trein'] = dados_trein
database['dados_valid'] = dados_valid
database['regras'] = regras
database['regras_classe'] = regras_classe
database['regras_CF'] = regras_CF

tempo = time.time() - tempo
print(tempo)

database.close()