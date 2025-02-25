import numpy as np
import pandas as pd

#Métricas de estrategia/benchmark
def metrica_cartera (serie_estrategia, serie_benchmark):
    '''
    A partir de las equity de la estrategia y del benchmark,
    se calcula el retorno anualizado de la estrategia y benchmark
    y ratios de sharpe de la estrategia y benchmark
    
    '''

    serie_estrategia = serie_estrategia.dropna()
    serie_benchmark = serie_benchmark.dropna()
    
    init_date, end_date = serie_benchmark.index[0], serie_benchmark.index[-1]
    years_in = (end_date - init_date) / pd.Timedelta(days=365, hours=6) # Los años duran 365 días y 6 horas
    bdays_year = int(serie_benchmark.shape[0]/years_in)
    
    total_estrategia = (serie_estrategia.iloc[-1]/serie_estrategia.iloc[0] - 1)
    ann_estrategia = np.power(total_estrategia + 1, 1/years_in) - 1
    total_benchmark = serie_benchmark.iloc[-1]/serie_benchmark.iloc[0] - 1
    ann_benchmark = np.power(total_benchmark + 1, 1/years_in) - 1
    
    logret_estrategia = np.log(serie_estrategia).diff().dropna()
    logret_benchmark = np.log(serie_benchmark).diff().dropna()
    riesgo_estrategia = logret_estrategia.var() 
    rent_estrategia = logret_estrategia.sum()/len(logret_estrategia)
    ann_sharpe_estrategia = ((rent_estrategia*252)/(np.sqrt(riesgo_estrategia)*np.sqrt(252)))
    
    riesgo_benchmark = logret_benchmark.var() 
    rent_benchmark = logret_benchmark.sum()/len(logret_benchmark)
    ann_sharpe_benchmark = ((rent_benchmark*252)/(np.sqrt(riesgo_benchmark)*np.sqrt(252)))
        

    return (ann_estrategia,
                ann_benchmark,
                ann_sharpe_estrategia,
                ann_sharpe_benchmark)

# Cálculo máximo drawdown
def drwdown (serie):
    '''
    A partir de la equity, se calcula su drawdown máximo
    
    '''
    
    dwserie = (serie - serie.expanding().max())/serie.expanding().max()
    maxdwn = np.round(dwserie.min()*-1,2)
    return (maxdwn, dwserie)

#Cálculo del alpha para ventanas roladas
def f_alpharol(s, b):
    '''
    Se calcula el alpha como la diferencia de la suma de los retornos
    logarítmicos del periodo
    
    '''
    
    w_b = b.loc[s.index]
    srets = np.log(s).diff().fillna(0)
    brets = np.log(w_b).diff().fillna(0)
    rent_srets = srets.sum()
    rent_brets = brets.sum()
    return rent_srets -  rent_brets

# Calcula parámetros para normalizar datos a con la distribución normal [0,1]
def fit_zscore01 (array):
    '''
    Se calculan los parámetros media, desviación estandar, mínimo y máximo del 
    array normalizado para posterior normalización/desnormalización
    
    '''
    std_dev = np.std(np.std(array, axis=0))
    media = np.mean(array)
    std_dev = np.std(array)
    array_normalizado = (array - media) / std_dev

    array_min  = np.min(array_normalizado)
    array_max  = np.max(array_normalizado)

    return media, std_dev, array_min, array_max

# Normaliza datos a [0,1]
def norm_zscore01(array, media, std_dev, array_min, array_max):
    '''
    Se normaliza el array de datos a partir de los parámetros
    generados por fit_zscore01
    
    '''

    array_normalizado = (array - media) / std_dev
    array_reescalado = (array_normalizado - array_min) / (array_max - array_min)

    return array_reescalado

# Desnormaliza datos
def desnorm_zscore01(array_normalizado, media, std_dev, array_min, array_max):
    '''
    Se desnormaliza el array de datos a partir de los parámetros
    generados por fit_zscore01
    
    '''
    
    # Revertir el escalado entre 0 y 1
    array_normalizado = array_normalizado * (array_max - array_min) + array_min

    # Revertir el Z-score
    array_original = array_normalizado * std_dev + media

    return array_original
