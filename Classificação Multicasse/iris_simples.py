import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import numpy as np

## transforma a classe de Strings pra numérico
labelEncoder = LabelEncoder()

## faz a coleta dos dados
base = pd.read_csv('original.csv')

## representa os dados utilizados para previsao
previsores = base.iloc[:, 0:4].values

# representa os resultados da previsão (nome das classes)
classe = base.iloc[:, 4].values

## converte efetivamente as strings para inteiros
classe = labelEncoder.fit_transform(classe)

## como a saída possui 3 neurônios, é necessário mapear a saída
## para uma matriz de conversão
## ex.: 
## iris setosa: 1 0 0 
## iris virginica : 0 1 0
## iris versicolor: 0 0 1
classe_dummy = np_utils.to_categorical(classe)

# define cada uma das classes de previsão e teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

## cria a classe classificadora e adiciona os neurônios em cada uma das camadas escondidas
classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classificador.add(Dense(units = 4, activation = 'relu'))

## 3 classes = 3 units
## softmax: semelhante ao sigmoide porém para mais de uma classe
## retorna a probabilidade de cada uma das classes de saída
classificador.add(Dense(units = 3, activation = 'softmax')) 
    
## compila para a execução 
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

# executa o treinamento
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 1000)   

## avalia o resultado obtido
resultado = classificador.evaluate(previsores_teste, classe_teste)

## preve os dados que ainda nao foram testados e os separa de acordo com o tamanho retornado
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

## faz a conversão para poder comparar os tipos de dados
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

## retorna a quantidade de acertos/erros pros testes realizados
matriz = confusion_matrix(previsoes2, classe_teste2)