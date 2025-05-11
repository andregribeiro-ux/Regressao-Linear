# Importando as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Ler a planilha do excel e substituir a "," por um "." (Se não o python não aceita)
dados = pd.read_excel(r"Coloque o caminho do seu arquivo que foi baixado aqui") 
dados['Vendas'].replace(",", ".")
dados['Dias'] = (dados['Data'] - dados['Data'].min()).dt.days #Fazendo uma nova coluna chamada dias

# Visualizando os valores máximos registrados
print(max(dados["Vendas"]))
print(min(dados["Vendas"]))

# Declarando as váriáveis de forma que o python suporte
X = dados['Dias'].values.reshape(-1, 1)
Y = dados['Vendas'].values

# Treinando o modelo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
modelo = LinearRegression()
modelo.fit(X_train, Y_train)
Y_pred = modelo.predict(X_test)

# Analisando a eficácia do modelo
r2 = modelo.score(X_test, Y_test)
print(f"R²:{r2:.2f} (Quanto mais próximo de 1, melhor)")

mse = mean_squared_error(Y_test, Y_pred)
print(f"MSE: {mse:.2f}")

# Fazendo uma previsão para o dia 1100
dia_1100 = np.array([[1100]])
vendas_previstas = modelo.predict(dia_1100)
print(f"{vendas_previstas[0]:.2f} Vendas previstas para o dia 1100")

# Exibindo os valores em um gráfico
plt.scatter(X_test, Y_test, color='blue', label='Real')
plt.plot(X_test, Y_pred, color='red', label='Predito')
plt.xlabel('Dias')
plt.ylabel('Vendas')
plt.legend()
plt.show()
