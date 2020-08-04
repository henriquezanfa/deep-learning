# Deep Learning de A a Z

## Funções de ativação

### Função Degrau

Caso o valor recebido seja maior ou igual a 1, retorna 1. Caso contrário, retorna 0.

### Função Sigmoid

Bastante usada para classificação de problemas binários, relacionadas a probabilidade.

![equation](https://latex.codecogs.com/svg.latex?y&space;=&space;\frac{1}{1&space;&plus;&space;e^{-x}})

### Função Tangente Hiperbólica

Bastante usada para classificação de problemas binários, porém assume valores negativos, variando de -1 a 1.

![equation](https://latex.codecogs.com/svg.latex?y&space;=&space;\frac{e^{x}-e^{-x}}{e^{x}&plus;e^{-x}})

### ReLU (Rectified Linear Units)

Caso o valor seja menos que zero, retorna zero. Caso contrário, retorna próprio valor.

![equation](https://latex.codecogs.com/svg.latex?y&space;=&space;max(0,&space;x))

### Função Linear

Retorna o próprio valor. Bastante utilizada em problemas de regressão.

![equation](https://latex.codecogs.com/svg.latex?y&space;=&space;x)

### Função Softmax

Usada para retornar probabilidades em problemas de mais de duas classes.

![equation](https://latex.codecogs.com/svg.latex?y&space;=&space;\frac{e^{x}}{\sum&space;e^{x}})