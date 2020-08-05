import numpy as np
import pandas as pd
from keras.models import model_from_json

# carregamento 
arquivo = open('classificador_cancer_mama.json', 'r')
estrutura = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura)
classificador.load_weights('classificador_cancer_mama.h5')

# testes com a rede carregada
novo = np.array([[17.99,	10.38,	122.8,	1010.0,	0.1584,	2.3776,	0.1001,	0.0471,	0.4419,	0.07871, 1295.0,	0.9053,	8589.0,	153.4,
                  0.006399	,0.04904,	0.053,	0.04587,	0.04003,	0.008193,	25.38,	17.33,	184.6,	2019.0,	
                  0.1622,	0.6656,	0.7119,	0.2654,	0.4601,	0.1189]])

previsao = classificador.predict(novo)

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')
classificador.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['binary_accuracy'])

resultado = classificador.evaluate(previsores, classe)
