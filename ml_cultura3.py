import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Importando dataset
NOMEARQUIVO = "Crop_Recommendation.csv"

def carregarDados(nomeArquivo):
    print("carregarDados")
    print("Nome do arquivo: %s" % nomeArquivo)
    dataFrame = pd.read_csv(nomeArquivo, delimiter=",")
    return dataFrame

def preparacaoDados(dataFrame):
    print("preparacaoDados")

    # Informações sobre o DataFrame
    print(dataFrame.info())

    # Primeiras linhas
    print(dataFrame.head())

    # Últimas linhas
    print(dataFrame.tail())

    # Quantidade de linhas e colunas
    print(dataFrame.shape)

    # Dados estatísticos básicos
    print(dataFrame.describe().T)

    # Verificando dados nulos
    print(dataFrame.isnull().sum())

    # Verificando dados duplicados
    print(dataFrame.duplicated().sum())

    return dataFrame

def treinarModelo(dataFrame):
    # Identificando as colunas de características e a coluna alvo
    colunas_caracteristicas = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
    coluna_alvo = 'Crop'

    # Separando features e target
    X = dataFrame[colunas_caracteristicas]
    y = dataFrame[coluna_alvo]

    # Dividindo os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinando o modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # Fazendo previsões
    y_pred = modelo.predict(X_test)

    # Avaliando o modelo
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Plotando a matriz de confusão
    plotarMatrizConfusao(y_test, y_pred, modelo.classes_) ## modelo.classe?

    # Plotando a importância das características
    plotarImportanciaCaracteristicas(X.columns, modelo.feature_importances_)# modelo.feature_importances?

    # Plotando o boxplot da proporção de chuva por cultura
    culturas = ['Rice', 'Maize', 'Coffee']  
    plotarBoxplotChuvaPorCultura(dataFrame, culturas)

def plotarMatrizConfusao(y_test, y_pred, classes):
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.show()

def plotarImportanciaCaracteristicas(feature_names, importances):
    # Avaliando a importância das características
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print(feature_importances)

    # Plotando a importância das características
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('Importância das Características')
    plt.xlabel('Importância')
    plt.ylabel('Características')
    plt.show()

def plotarBoxplotChuvaPorCultura(dataFrame, culturas):
    # Filtrando os dados para as culturas de interesse
    dataFiltrada = dataFrame[dataFrame['Crop'].isin(culturas)]
    
    # Plotando o boxplot para a característica Rainfall em relação às culturas
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Crop', y='Rainfall', data=dataFiltrada, order=culturas, palette='Set1')
    plt.title('Distribuição da Proporção de Chuva por Cultura')
    plt.xlabel('Cultura')
    plt.ylabel('Proporção de Chuva')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dataFrame = carregarDados(NOMEARQUIVO)
    dataFrame = preparacaoDados(dataFrame)
    treinarModelo(dataFrame)
    

    
