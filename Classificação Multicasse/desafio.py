import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

## transforma a classe de Strings pra numérico
labelEncoder = LabelEncoder()

base = pd.read_csv('original.csv')

## representa os dados utilizados para previsao
previsores = base.iloc[:, 0:4].values

# representa os resultados da previsão (nome das classes)
classe = base.iloc[:, 4].values

classe = labelEncoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede(optimizer, kernel_initializer, activation, neurons, dropout):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, input_dim = 4))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units = 4, activation = 'relu'))
    
    ## 3 classes = 3 units
    ## softmax: semelhante ao sigmoide porém para mais de uma classe
    ## retorna a probabilidade de cada uma das classes de saída
    classificador.add(Dense(units = 3, activation = 'softmax')) 
        
    ## compila para a execução 
    classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criar_rede)

parametros = {'batch_size': [10, 30],
              'epochs': [2000, 3000],
              'dropout': [0.2, 0.3],
              'optimizer': ['adam', 'sgd'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh', 'sigmoid'],
              'neurons': [4, 8, 16]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,                           
                           cv = 2)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_


## grava classificador json
classificador_json = classificador.to_json()
with open('classificador.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador.h5')

## le classificador json
arquivo = open('classificador.json', 'r')
estrutura = arquivo.read()
arquivo.close()
classificador_carregado = model_from_json(estrutura)
classificador_carregado.load_weights('classificador.h5')
    
## novo registro
reg = np.array([[3.2, 4.5, 0.9, 1.1]])
previsao = classificador.predict(reg)
previsao = (previsao > 0.5)
if previsao[0][0] == True and previsao[0][1] == False and previsao[0][2] == False:
    print('Iris setosa')
elif previsao[0][0] == False and previsao[0][1] == True and previsao[0][2] == False:
    print('Iris virginica')
elif previsao[0][0] == False and previsao[0][1] == False and previsao[0][2] == True:
    print('Iris versicolor')

