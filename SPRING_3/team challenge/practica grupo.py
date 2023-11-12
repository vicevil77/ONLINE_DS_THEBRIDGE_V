# IMPORTS
#VARIABLES Y FUNCIONES
#DEFINICION DE CLASES
# CODIGO PRINCIPALç
from sys import path
path.append(".\\")
import numpy as np


import funciones as fn

tam_tablero = input (" que tamaño quieres? ancho , alto")

if  tam_tablero != "":

    lista_dimensiones =[int(elemento) for elemento in tam_tablero.split(",")]
else:
    lista_dimensiones = fn.TAM_DEFECTO, fn.TAM_DEFECTO

tablero = fn.crea.tablero ((lista_dimensiones))

fn.coloca_barco(tablero, [(1,1), (2,2),(1,3)])


print(tablero)