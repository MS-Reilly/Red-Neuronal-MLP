# Para la construcción del codigo del Perceptron multicapa
# Necesitaremos importar dos bibliotecas como se muestra a continuación
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():

    def __init__(self,i = 1):
        """Inicializar la función"""

        # Se inician unos valores aleatorios fijos para poder ralizar el modelo
        np.random.seed(i)
        self.synaptic_weights = np.random.random((3,1)) *2 -1 #(el rango de valores es de 0,1 por lo que se multiplica por 2 y se le resta 1 para normalizarlo )


    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))


    def sigmoid_derivative(self, x):
        return x * (1-x)


    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):

            output = self.think(training_inputs)
            print(f'outpus is {output}')
            error = training_outputs - output
            print(f'\n el error es {error}')
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):

            inputs = inputs.astype(float)
            print(f'\n{inputs}, \n ------')
            output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

            return output



if __name__ == "__main__":

    neural_network = NeuralNetwork()
    print("random synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[1,0,1],[1,0,0],[0,1,1]])

    training_outputs = np.array([[1,1,0]]).T
    #print(training_outputs)
    neural_network.train(training_inputs, training_outputs, 3)

    #print("Synaptic weights after training: ")
    #print(neural_network.synaptic_weights)

    #A = str(input("Input 1: "))
    #B = str(input("Input 2: "))
    #C = str(input("Input 3: "))

    #print("New situation: input data = ", A , B, C)
    #print("Output Data: ")
    #print(neural_network.think(np.array([A,B,C])))


    # 1 la tasa se preve hawkish
    # 0 La tasa es neutra
    # -1 la tasa es dovish
