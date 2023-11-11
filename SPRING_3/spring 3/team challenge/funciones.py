import numpy as np


TAM_DEFECTO  = 15


def crea_tablero(dimensiones):
    return np.full(dimensiones, " ")

def coloca_barco(tablero,barco):
    for pieza in barco:
        tablero[pieza] = "0"


if __name__ == "__main__":
    print(crea_tablero((15,10)))