import numpy as np

def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))

def tahnFunction(soma):
    return ((np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma)))

def reluFunction(soma):
    if(soma < 0):
        return 0
    return soma

def linearFunction(soma):
    return soma

# deve retornar 1
testeStepFunction = stepFunction(5)
# deve retornar 0
testeStepFunction = stepFunction(-5)

# deve retornar 0.9933071490757153
testeSigmoidFunction = sigmoidFunction(5)
# deve retornar 0.0066928509242848554
testeSigmoidFunction = sigmoidFunction(-5)

# deve retornar 0.999909204262595
testeTahnFunction = tahnFunction(5)
# deve retornar -0.999909204262595
testeTahnFunction = tahnFunction(-5)

# deve retornar 5
testeReluFunction = reluFunction(5)
# deve retornar 0
testeReluFunction = reluFunction(-5)

# deve retornar 5
testeLinearFunction = linearFunction(5)
# deve retornar -5
testeLinearFunction = linearFunction(-5)