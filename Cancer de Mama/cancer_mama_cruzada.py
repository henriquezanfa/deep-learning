import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede():
    classificador = Sequential()
    
    classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer='random_uniform', input_dim = 30))
    # zera o valor de 20% dos neuronios da camada
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer='random_uniform'))
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = 1, activation= 'sigmoid'))

    otimizador = keras.optimizers.Adam(lr = 0.001, decay=0.0001)
    
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics= ['binary_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn=criarRede, 
                                epochs = 100, 
                                batch_size = 10)

# cv => numero de validações cruzadas
resultados = cross_val_score(estimator = classificador, 
                             X = previsores,
                             y = classe, 
                             cv = 10, 
                             scoring='accuracy')

# media dos resultados da rede
media = resultados.mean()

# desvio padrao
desvio = resultados.std()
