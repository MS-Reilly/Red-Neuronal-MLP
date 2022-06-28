# Se importan las diferentes bibliotecas que nos ayudaran a modelar la red neuronal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Se crea una clase para definir todo el funcionamiento de la red neuronal

class NeuralNetwork():

    def __init__(self, documento):
        """ Inicializar los attributos y los pesos"""

        # La siguiente parte del codigo es la encargada de importar los datos del documento CSV
        # Posteriormente imprimimos la especificacion de los datos, lo cual nos ayuda a programar despues.
        self.datos = pd.read_csv(documento)
        np.random.seed(1)


        self.synaptic_weights = 2 * np.random.random((10,1))-1

    def sigmoid(self, x):

        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):

        return x * (1-x)

    def data(self,label):
        """Describimos los inputs y outputs"""
        # Separamos los datos en dos categorias el output el cual le pondremos Label
        # y nuestras variables dependientes las cuales le pondremos features

        self.labels = np.array(self.datos[label], dtype=np.float128)
        self.features = np.array(self.datos.drop(columns = [label]),dtype=np.float128)

        #print(self.labels)
        print(self.labels)
        # Para poder utilizar los numero con dijitos para nuestra neurona
        #los debemos de clasificar como floats

        #self.features = self.features.values.astype('float32')
        #self.labels = self.labels.values.astype('float32')


    def split_data (self, size = 0.2):

        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(self.features, self.labels, test_size = size)


    def train(self, training_iterations = 1000):

        for iteration in range(training_iterations):

            self.labels = self.think(self.features_train)
            self.error = self.labels_train - self.labels
            self.adjustments = np.dot(self.features_train.T, self.error * self.sigmoid_derivative(self.labels))
            self.synaptic_weights += self.adjustments

    def think(self, inputs):

        inputs = inputs.astype(float)
        x = np.dot(inputs, self.synaptic_weights)
        self.labels = self.sigmoid(x)

        return self.labels



    def prediction(self):

        A = str(input("Inflación no-subyacente: "))
        B = str(input("Inflación subyacente: "))
        C = str(input("Brecha del Producto: "))
        D = str(input("Brecha del Empleo: "))
        E = str(input("Brecha de la Tasa de referencia: "))


        print("Por favor ingrese los siguientes datos como decimales: ", A , B, C, D, E)
        print("La inflación esperada dados los datos anteriores es de: ")
        print(self.think(np.array([A,B,C,D])))



if __name__ == "__main__":

    inflacion = NeuralNetwork('prueba.csv')
    inflacion.data('Inflacion')
    #print(inflacion.synaptic_weights)
    inflacion.split_data()
    inflacion.train()
    #print(inflacion.synaptic_weights)
    #inflacion.prediction()
