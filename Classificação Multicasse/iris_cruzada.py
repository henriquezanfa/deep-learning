import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score 
from sklearn.preprocessing import LabelEncoder

## transforma a classe de Strings pra numérico
labelEncoder = LabelEncoder()

base = pd.read_csv('original.csv')

## representa os dados utilizados para previsao
previsores = base.iloc[:, 0:4].values

# representa os resultados da previsão (nome das classes)
classe = base.iloc[:, 4].values

classe = labelEncoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede():
    classificador = Sequential()
    classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
    classificador.add(Dense(units = 4, activation = 'relu'))
    
    ## 3 classes = 3 units
    ## softmax: semelhante ao sigmoide porém para mais de uma classe
    ## retorna a probabilidade de cada uma das classes de saída
    classificador.add(Dense(units = 3, activation = 'softmax')) 
        
    ## compila para a execução 
    classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criar_rede, 
                                epochs = 1000,
                                batch_size = 10)
    
resultados = cross_val_score(estimator = classificador, X = previsores, y = classe, 
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()