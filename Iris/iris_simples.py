import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

base = pd.read_csv('iris.csv')

previsores = base.iloc[:,0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder

# transforma a classe que era String em numérico
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

# 1 0 0 - Iris-setosa
# 0 1 0 - Iris-versicolor
# 0 0 1 - Iris-setosa
classe_dummy = np_utils.to_categorical(classe)

from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size = 0.25)

classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dense(units = 3, activation = 'softmax'))

classificador.compile(optimizer = 'adam', 
                      loss = 'categorical_crossentropy', 
                      metrics = ['categorical_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs= 1000)

# faz a previsao utilizando a keras 
resultado = classificador.evaluate(previsores_teste, classe_teste)

# faz a previsao "manualmente" 
previsoes = classificador.predict(previsores_teste)
previsoes = previsoes > 0.5

# converte a saída o índice que queremos 
# 1 0 0 => retorna 0
# 0 1 0 => retorna 1
# 0 0 1 => retorna 2
import numpy as np
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

                 
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(previsoes2, classe_teste2)
