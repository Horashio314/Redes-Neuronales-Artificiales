# !/usr/bin/env python3
# coding: utf-8

# # Importar librerías
from pylab import rand, plot, show, norm

# # Clase Perceptron
class Perceptron:
    """
    Perceptron Simple

    Parametros:
        w_: array-1d
            Pesos actualizados después del ajuste.
            
        tasaApren_: float
            Tasa de aprendizaje.
    """

    def __init__(self, w_= [0, 0], tasaApren_ = 0.1):
        """ 
        Metodo constructor del preceptron, 
        inicialza los valores por defecto.
        """
        self.w = w_                         # Vector w, representa los pesos.
        self.tasaApren = tasaApren_         # Tasa de aprendizaje.

    def respuesta(self, x):
        """ 
        Salida del perceptron, aplica el producto punto entre w (pesos) y x (data).
        
        Parametros:
            x: list, forma [valor 1, valor 2]
                Data que se esta analizando.

        Retorna:
            int: Si el producto punto es mayor o igual a uno (1) devuelve '1' de lo contrario '0'
        """
        y = (x[0] * self.w[0]) + (x[1] * self.w[1])     # Producto punto entre w y x.
        
        if y >= 1:
            return 1
        else:
            return 0

    def actualizarPesos(self, x, error):
        """
        Metodo encargado de actualizar el valor de los pesos en el vector w:        
            w(t+1) = w(t) + (tasaApren * error * x)
            
            Donde:
                w(t+1): Es el peso para la siguiente iteracion de aprendizaje.
                w(t): Es el peso para la iteracion actual de aprendizaje.
                tasaApren: Tasa de aprendizaje.
                error: (respuesta deseada) - (respuesta del perceptron).
                x: Valor actual.

        Parametros:
            x: list, forma [valor 1, valor 2]
                Data que se esta analizando.
        """
        self.w[0] += self.tasaApren * error * x[0]
        self.w[1] += self.tasaApren * error * x[1]

    def entrenamiento(self, data):
        """
        Metodo encargado de entrenar el perceptron simple, el vector en los datos, cada vector en los datos debe tener 3 elementos,
        el tercer elemento (x[2]) debe ser etiquetado (salida deseada)

        Parametros:
            data: list, forma [[x1, y1, resp1], [x2, y2, resp2], ... , [xn, yn, respn]]
                Vector con los datos, cada uno debe tener la forma, valor 1, valor 2 y respuesta deseada.
        """
        aprendio = False                        # Determina si el perceptron aprendio segun el criterio.
        iteracion = 0                           # Nunero de iteracion que le tomo al perceptron aprender.

        while not aprendio:                     # Mientras no aprenda.
            globalError = 0.0                   # Mantiene el error general que se va obteniendo el aprendizaje.            

            for x in data:                      # Recorremos los datos.
                r = self.respuesta(x)           # Obtenemos la respuesta del perceptron sobre dato.

                if x[2] != r:                           # Si la respuesta no es la deseada.
                    error = x[2] - r                    # El error en la iteracion se actualiza a: respuesta deseada - respuesta obtenida.
                    self.actualizarPesos(x, error)      # Se actualiza los pesos con el dato y el error de la iteracion.
                    globalError += abs(error)           # Se actualiza el error general del perceptron.

            iteracion += 1                              # Se contabiliza la iteracion para el criterio de aprendizaje.
            
            if globalError == 0.0 or iteracion >= 1000:      # Criterio de salida: si el error general es 0, o la iteracion de apredndizaje sobre el 100.
                aprendio = True                             # Salida del perceptron.

def datosGenerados():
    """
    Metodo encargado de generar un conjunto de datos de prueba, linealmente separables, 
    con la siguiente forma:
        [[x1, y1, resp1], [x2, y2, resp2], ... , [xn, yn, respn]]

        Donde:
            xn: Representa el valor 1.
            yn: Representa el valor 2.
            respn: Representa la etiqueta de la muestra.

    Retorna:
        list: Lista con los datos con la siguiente forma: 
            [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
    """

    datos = []
    datos.append([0,0,0])
    datos.append([0,1,0])
    datos.append([1,0,0])
    datos.append([1,1,1])

    return datos

if __name__ == "__main__":
    datosEntrenamiento = datosGenerados()               # Se genera los datos de prueba con los que entrenara el perceptron.
    perceptron = Perceptron()                           # Se instancia del perceptron.
    perceptron.entrenamiento(datosEntrenamiento)        # Se entrena el perceptron con los datos de prueba.
    datosPrueba = datosGenerados()                      # Se genera los datos con los que probara el perceptron.

    # Se prueba el perceptron con los datos de prueba.
    for x in datosPrueba:                               # Se recorre los datos de prueba.
        r = perceptron.respuesta(x)                     # Obtenemos la respuesta del perceptron.
    
        if r != x[2]:                                   # Verificamos si la respuesta no es correcta.  
            print ('error')                             # Si no es correcta, imprimimos 'error', no se agrega el punto a la grafica.
        else:                                      
            print("[",x[0],",",x[1],"] -----> ", x[2])      # De lo contrario, la respuesta es correcta y se imprime.
