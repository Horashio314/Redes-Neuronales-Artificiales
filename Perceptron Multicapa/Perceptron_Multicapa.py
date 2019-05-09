#!/usr/bin/python

import numpy as np

def sigmoid(x):
    """
    Metodo encargado de retorna la resultado 
    de la funcion Sigmoid evaluada en x.
    
    Parametros:
        n: int
            Valor en la que se evaluara la función.

    Retorna:
        int: Funcion Sigmoid evaluada en x.
    """
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_der(x):
    """
    Metodo encargado de retorna la resultado de la 
    derivada de la funcion Sigmoid evaluada en x.
    
    Parametros:
        n: int
            Valor en la que se evaluara la función.

    Retorna:
        int: Derivada de la funcion 
            Sigmoid evaluada en x.
    """
    return x*(1.0 - x)

class Multicapa:
    def __init__(self, entrada):
        self.entrada = entrada
        self.l=len(self.entrada)
        self.li=len(self.entrada[0])

        self.wi=np.random.random((self.li, self.l))
        self.wh=np.random.random((self.l, 1))

    def think(self, inp):
        s1=sigmoid(np.dot(inp, self.wi))
        s2=sigmoid(np.dot(s1, self.wh))
        return s2

    def train(self, entrada, salidas, it):
        for i in range(it):
            l0=entrada
            l1=sigmoid(np.dot(l0, self.wi))
            l2=sigmoid(np.dot(l1, self.wh))

            l2_err=salidas - l2
            l2_delta = np.multiply(l2_err, sigmoid_der(l2))

            l1_err=np.dot(l2_delta, self.wh.T)
            l1_delta=np.multiply(l1_err, sigmoid_der(l1))

            self.wh+=np.dot(l1.T, l2_delta)
            self.wi+=np.dot(l0.T, l1_delta)

if __name__ == "__main__":
    entrada=np.array([[0,0], [0,1], [1,0], [1,1] ])
    salidas=np.array([ [0], [1],[1],[0] ])
    
    n=Multicapa(entrada)
    print(n.think(entrada))
    print("")
    n.train(entrada, salidas, 1000000)
    print(n.think(entrada))
