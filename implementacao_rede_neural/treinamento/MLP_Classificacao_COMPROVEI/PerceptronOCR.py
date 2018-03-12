# -*- coding:utf-8 -*-
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from datetime import datetime

hora_i = datetime.now()
print(hora_i)

df = pd.read_csv('dados-canhoto.csv', header=None)

# saida
y_train = df.iloc[0:100, [39]].values
print('Saídas esperadas:')
print(y_train)

# entrada
X_train = df.iloc[0:100, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]].values
print('Dados originais:')
print(X_train)

# Teste saida
y_Teste = df.iloc[101:266, [39]].values
print('Teste saídas:')
print(y_Teste)

# Teste entrada
X_Teste = df.iloc[101:266, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]].values
print('Teste dados:')
print(X_Teste)

mlp = MLPClassifier(hidden_layer_sizes=(3,1),
                    max_iter=1000,
                    alpha=1e-1,
                    solver='lbfgs', # Para pequenos conjuntos de dados o 'lbfgs' pode convergir mais rápido e um melhor desempenho.
                    verbose=True,
                    tol=1e-9, # Tolerância para a otimização.
                    )
#
# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-6, random_state=1, learning_rate_init=.0003,)
#
# mlp = MLPClassifier(hidden_layer_sizes=(7,), max_iter=1000, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-7, random_state=1,
#                     learning_rate_init=.1)


mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_Teste, y_Teste))

hora_f = datetime.now()
print(hora_f)
print(hora_f-hora_i)
