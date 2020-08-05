import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

classificador = Sequential()

classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'normal', input_dim = 30))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 1, activation= 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['binary_accuracy'])
    
classificador.fit(previsores, classe, batch_size = 10, epochs= 100)


novo = np.array([[17.99,	10.38,	122.8,	1010.0,	0.1584,	2.3776,	0.1001,	0.0471,	0.4419,	0.07871, 1295.0,	0.9053,	8589.0,	153.4,
                  0.006399	,0.04904,	0.053,	0.04587,	0.04003,	0.008193,	25.38,	17.33,	184.6,	2019.0,	
                  0.1622,	0.6656,	0.7119,	0.2654,	0.4601,	0.1189]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)
