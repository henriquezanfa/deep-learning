# importação dos arquivos
import pandas as pd
previsores = pd.read_csv("entradas_breast.csv")
classe = pd.read_csv("saidas_breast.csv")

# divisão dos dados entre teste (25%) e treinamento (75%)
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25)

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

# segunda camada oculta
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer='random_uniform'))

# camada de saída
classificador.add(Dense(units = 1, activation= 'sigmoid'))

# lr => variação prao calculo do erro
# decay => diminuição do lr a cada atualização
otimizador = keras.optimizers.Adam(lr = 0.001, decay=0.0001)

# optimizer => função de ajuste dos pesos
# loss => calculo do erro
# metrics => classificação dos resultados
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics= ['binary_accuracy'])

# batch_size => numero de calculo de erros para recalcular os pesos
# epochs => numero de ajustes de peso
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)


pesos0 = classificador.layers[0].get_weights()
# pesos0 tem tamanho = 2. 
# O primeiro valor representa os valores de entrada do problema.
# O segundo valor representa o Bias (neuronio adicionado por default na primeira camada)

pesos1 = classificador.layers[1].get_weights()

pesos2 = classificador.layers[2].get_weights()


previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)

matriz = confusion_matrix(classe_teste, previsoes)


resultado = classificador.evaluate(previsores_teste, classe_teste)
