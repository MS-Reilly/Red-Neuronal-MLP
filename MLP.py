#Para poder construir el modelo nos apoyaremos las siguientes bibliotecas
import numpy as np # numpy nos ayudara a realizar diferentes operaciones matriciales
import matplotlib.pyplot as plt # noas ayudara a graficar diferentes cosas
import pandas as pd # nos ayudara a leer los documentos
import tensorflow as tf #Deep Learning que nos ayuda a programar de manera mas eficiente la red
from tensorflow import keras #Una biblioteca diseñanda para realizar redes neuronales secuenciales y densas
from sklearn.model_selection import train_test_split # esta función nos ayudara a dividir los datos en un tamaño programado



# Para iniciar la construcción del modelo iniciaremos creando una clase
#la cual nombraremos MLP

class MLP():

    #Definiremos los parametros que deben de iniciarse en primera instancea
    #para esto definimos una función de apoyo

    def __init__(self,inputs_dim, neuronas_capa, output_dim):
        """Inicializamos la red neuronal"""
        # Identificamos el nuemro de inputs, neuronas por capa y la dimensión del output
        self.n_inputs = inputs_dim
        self.n_capas = neuronas_capa
        self.n_salidas = output_dim

    def pesos_aleatorios(self,i):
        """Iniciar pesos sinapticos aleatorios standard"""
        # se generan pesos aleatorios que sean standarizados en cada prueba
        np.random.seed(i)

    def cargar_datos (self, file_path, size= 0.1):
        """Se leen los datos y se introducen a un array"""
        # Se utiliza la función de Pandas para leer el documento
        self.data = np.loadtxt(file_path, delimiter =',')
        self.inputs = self.data[:,0:5]
        self.outputs = self.data[:,5]
        print(self.inputs[0])
        print(self.outputs[0])

        # Dividimos los datos
        self.inputs_train, self.inputs_test, self.outputs_train, self.outputs_test = train_test_split(self.inputs, self.outputs, test_size = size)

    def feed_foward (self):
        """Introducimos nuestros datos a la capa de entrada"""

        #definimos el modelo inicial como uno secuancial
        self.MLP = keras.Sequential()

        # le agregamos las tres capas ocultas
        self.MLP.add(keras.layers.Dense(10,input_dim = 5, activation = 'relu'))
        self.MLP.add(keras.layers.Dense(10, activation = 'relu'))
        self.MLP.add(keras.layers.Dense(10, activation = 'relu'))

        # le agregamos nuestra capa de output
        self.MLP.add(keras.layers.Dense(1))

    def compilar_modelo (self):

        self.MLP.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = ['accuracy'])

    def entrenar_modelo(self):
        self.MLP.fit(self.inputs_train,self.outputs_train, epochs = 1000, batch_size = 32)

    def evaluar_modelo(self):
        print("\nEvaluación del Modelo: \n")
        self.inflacion = self.MLP.evaluate(self.inputs_test, self.outputs_test)
        print("\n%s: %.2f%%" % (self.MLP.metrics_names[1], self.inflacion[1]*100))

        self.MLP.summary()

    def predecir_modelo(self):
        """Dados unos inputs predice el modelo"""
        X_nueva = self.inputs_test[0:3]
        Y_predecida = self.MLP.predict(X_nueva)
        print(f'La prediccion del modelo es {Y_predecida}')




if __name__ == "__main__":

    neural = MLP(5,10,1)
    neural.pesos_aleatorios(3)
    neural.cargar_datos('datos_red123.csv')
    neural.feed_foward()
    neural.compilar_modelo()
    neural.entrenar_modelo()
    neural.evaluar_modelo()
    neural.predecir_modelo()
