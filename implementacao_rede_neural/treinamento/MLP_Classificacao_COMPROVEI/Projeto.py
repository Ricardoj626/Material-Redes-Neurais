# -*- coding:utf-8 -*-
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

df = pd.read_csv('pima-indians-diabetes.csv', header=None)

# saida
y_train = df.iloc[0:1000, [8]].values
print('Saídas esperadas:')
print(y_train)

# entrada
X_train = df.iloc[0:1000, [0, 1, 2, 3, 4, 5, 6, 7]].values
print('Dados originais:')
print(X_train)

# Teste saida
y_Teste = df.iloc[1000:1145, [8]].values
print('Teste saídas:')
print(y_Teste)

# Teste entrada
X_Teste = df.iloc[1000:1145, [0, 1, 2, 3, 4, 5, 6, 7]].values
print('Teste dados:')
print(X_Teste)

mlp = MLPClassifier(hidden_layer_sizes=(7,3),
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
