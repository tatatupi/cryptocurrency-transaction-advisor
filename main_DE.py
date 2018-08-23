#import nebulosos as neb
import genetico
import shelve 
import numpy as np
import time
import random
import pandas as pd
from deap import base, creator, tools

## Lendos os Dados 

resultados = shelve.open('results_100-50', flag='c')

database = shelve.open('all_data', flag='r') 

#variáveis da base de dados
taxa = database['taxa']
data = database['data']
entrada = database['entrada']
variaveis_linguisticas = database['variaveis_linguisticas']
classes = database['classes']
dados_all = database['dados']
matriz_pertinencias_all = database['matriz_pertinencias']
saida_classificada_all = database['saida_classificada']
dados_trein_all = database['dados_trein']
dados_valid_all = database['dados_valid']
regras_all = database['regras']
regras_CF_all = database['regras_CF']
regras_classe_all = database['regras_classe']

database.close()

## Parâmetros do algoritmo genético
itens_low = 1
itens_high = 2
regras_low = 3
regras_high = 5
itens_regra_par = np.arange(itens_low,itens_high+1) #quantidade máxima de parâmetros por regra
n_regras_par = np.arange(regras_low,regras_high+1) #quantidade máxima de parâmetros por regra
n_pop = 100 # número de indivíduos na população
n_gen = 50
P_cruzamento = 0.6
P_mutacao = 0.35

CR = 0.25
F = 1  

## Inicializando as classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

## Variáveis com resultados
max_fits_all = pd.DataFrame(columns=entrada, index=range(n_gen+1))
max_fits_all.index.name = "Geração"

min_fits_all = pd.DataFrame(columns=entrada, index=range(n_gen+1))
min_fits_all.index.name = "Geração"

mean_fits_all = pd.DataFrame(columns=entrada, index=range(n_gen+1))
mean_fits_all.index.name = "Geração"

std_fits_all = pd.DataFrame(columns=entrada, index=range(n_gen+1))
std_fits_all.index.name = "Geração"

best_ind_all = {}

hof_all = {}


tempo = time.time()
## inicializando o loop das saídas
for saida in entrada:
    dados = dados_all[saida]
    matriz_pertinencias = matriz_pertinencias_all[saida]
    saida_classificada = saida_classificada_all[saida]
    dados_trein = dados_trein_all[saida]
    dados_valid = dados_valid_all[saida]
    regras = regras_all[saida]  
    regras_CF = regras_CF_all[saida]
    regras_classe = regras_classe_all[saida]

    ## Inicializando o DEAP    
    toolbox = base.Toolbox()
    toolbox.register("attr_ind", genetico.attr_individuo, regras)
    toolbox.register("individual", genetico.initPop, creator.Individual, n_regras_par, regras_classe, regras)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", genetico.calc_acerto)
    toolbox.register("select", tools.selRandom, k=3)
    
    ## Gerando a população inicial       
    population = [] 
    population = toolbox.population(n=n_pop)
    
    fits=[]
    for i in range(len(population)):
        fits.append(toolbox.evaluate(population[i], dados_valid, saida_classificada, classes, matriz_pertinencias, regras_classe, regras_CF, regras))
    
    for fits, ind in zip(fits, population):
        ind.fitness.values = fits
    
    best_ind = []
    hof = tools.HallOfFame(3)
    hof.update(population)
    best_ind.append(hof[0])
    
    offspring = []
    
    fits = [ind.fitness.values[0] for ind in population]
    length = len(population)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
    
    max_fits_all[saida][0] = max(fits)
    min_fits_all[saida][0] = min(fits)
    mean_fits_all[saida][0] = mean
    std_fits_all[saida][0] = std
    
    print("  Começando saída " + saida)
    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)
    print(" ")
    
    for gen in range(n_gen):
        
        
        for k, agent in enumerate(population):
            a,b,c = toolbox.select(population)
            y = toolbox.clone(agent)
            index = random.randrange(NDIM)
            for i, value in enumerate(agent):
                if i == index or random.random() < CR:
                    y[i] = a[i] + F*(b[i]-c[i])
            y.fitness.values = toolbox.evaluate(y)
            if y.fitness > agent.fitness:
                population[k] = y
        
     
        # Evaluate the new individuals
        for ind in offspring:
            if not ind.fitness.valid:  
                ind.fitness.values = toolbox.evaluate(ind, dados_valid, saida_classificada, classes, matriz_pertinencias, regras_classe, regras_CF, regras)
        
        population = toolbox.select(offspring, k=len(population))
        
        hof.update(population)
        best_ind.append(hof[0])
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]
    
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        max_fits_all[saida][gen+1] = max(fits)
        min_fits_all[saida][gen+1] = min(fits)
        mean_fits_all[saida][gen+1] = mean
        std_fits_all[saida][gen+1] = std
    
    print("  Resultado saída " + saida)
    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)
    print(" ")
    
    best_ind_all[saida] = best_ind
    hof_all[saida] = hof

tempo = time.time() - tempo
print(tempo)

resultados['max_fits'] = max_fits_all
resultados['min_fits'] = min_fits_all
resultados['mean_fits'] = mean_fits_all
resultados['std_fits'] = std_fits_all
resultados['best_ind'] = best_ind_all
resultados['hof'] = hof_all 
resultados['tempo'] = tempo

resultados.close()