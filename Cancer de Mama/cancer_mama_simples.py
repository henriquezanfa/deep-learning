# importação dos arquivos
import pandas as pd
previsores = pd.read_csv("entradas_breast.csv")
classe = pd.read_csv("saidas_breast.csv")

# divisão dos dados entre teste (25%) e treinamento (75%)
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, clases_teste = train_test_split(previsores, classe, test_size = 0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense # camada densa: cada um dos neurônios é ligado a todos os neurônios da camada subsequente

# criação da rede neural
classificador = Sequential()

# units => quantidade de neurônios na camada (fórmula: (#neuronios na camada de entrada + #neuronios na camada de saida) / 2)
# activation => função de ativação
# kernel_initializer => valor inicial dos pesos
# input_dim => número de entradas

# primeira camada oculta
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer='random_uniform', input_dim = 30))

# camada de saída
classificador.add(Dense(units = 1, activation= 'sigmoid'))

# optimizer => função de ajuste dos pesos
# loss => calculo do erro
# metrics => classificação dos resultados
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['binary_accuracy'])

# batch_size => numero de calculo de erros para recalcular os pesos
# epochs => numero de ajustes de peso
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)
