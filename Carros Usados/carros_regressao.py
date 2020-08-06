import pandas as pd

base = pd.read_csv('carros.csv', encoding='ISO-8859-1')
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)

# Aqui verificamos a variabilidade de dados nessas colunas 
# pra entender se vale a pena ou não mante-las
base['name'].value_counts()
base = base.drop('name', axis = 1)

base['seller'].value_counts()
base = base.drop('seller', axis = 1)

base['offerType'].value_counts()
base = base.drop('offerType', axis = 1)

# Verificamos a quantidade de preços abaixo de 10 e removemos do dataframe
precos_inconsistentes_baixo = base.loc[base.price <= 10]
base = base.loc[base.price > 10]

# Verificamos a quantidade de preços acima de 500000 e removemos do dataframe
precos_inconsistentes_cima = base.loc[base.price > 500000]
base = base.loc[base.price < 500000]

# Setando as variáveis nulas com o valor que mais aparece 

base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() # limousine

base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() # manuell

base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() # manuell

base.loc[pd.isnull(base['model'])]
base['model'].value_counts() # golf

base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() # benzin

base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() # nein

valores = {'vehicleType': 'limousine', 'gearbox' : 'manuell', 'model' : 'golf', 'fuelType' : 'benzin', 'notRepairedDamage' : 'nein'}

base = base.fillna(value = valores)
