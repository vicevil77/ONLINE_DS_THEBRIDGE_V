# PROGRAMA GEOLOCALIZAR IPS DE UN DF
import pandas as pd
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut  
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Point
import time

# abrimos el dataframe
archivos_xlsx = r'E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\DATASETS\DATASETS MODIFICADOS\ciber_modificado.xlsx'
df_geo = pd.read_excel(archivos_xlsx, engine='openpyxl')

# Crea una instancia del geocodificador (utilizando Nominatim en este caso).
geolocator = Nominatim(user_agent="my_geocoder")

# Crea una lista de las direcciones IP de origen y destino a partir del marco de datos
IP_origen = df_geo['IP Origen'].tolist()
IP_destino = df_geo['Destino IP'].tolist()

# Crea una lista vacía para almacenar las coordenadas de cada dirección IP
coordenadas = []

# Función de reintento para la geocodificación con un máximo de 3 intentos.
def geolocalizacion_IPs(localizacion):
    intentos = 0
    while intentos < 3:
        try:
            return geolocator.geocode(localizacion)
        except (GeocoderTimedOut, TimeoutError) as e:
            print(f"Error geocoding {localizacion}: {e}")
            intentos += 1
            time.sleep(1)  #Agrega una breve pausa antes de volver a intentarlo
    return None

# Itera a través de la lista de direcciones IP de origen y destino y obtén las coordenadas para cada una.
for source_ip, destination_ip in zip(IP_origen, IP_destino):
    try:
        ubicacion_origen = geolocator.geocode(source_ip, timeout=10)
        ubicacion_destino = geolocator.geocode(destination_ip, timeout=10)
        
        # Si ambas ubicaciones se encuentran, se añaden a las coordenadas
        if ubicacion_origen and ubicacion_destino:
            coordenadas.append((ubicacion_origen.longitude, ubicacion_origen.latitude, ubicacion_destino.longitude, ubicacion_destino.latitude))
        else:
            coordenadas.append((None, None, None, None))  # si la ubicación no se encuentra
    except TimeoutError:
        # Manejar excepciones, por ejemplo, añadir un retraso y reintento
        time.sleep(1)  # Espera 1 segundo antes de volver a intentar para evitar colapasar con peticiones
        try:
            ubicacion_origen = geolocator.geocode(source_ip, timeout=10)
            ubicacion_destino = geolocator.geocode(destination_ip, timeout=10)
            
            if ubicacion_origen and ubicacion_destino:
                coordenadas.append((ubicacion_origen.longitude, ubicacion_origen.latitude, ubicacion_destino.longitude, ubicacion_destino.latitude))
            else:
                coordenadas.append((None, None, None, None))
        except TimeoutError:
            coordenadas.append((None, None, None, None))  # Si aún hay un error, añade None

# Crear un GeoDataFrame a partir de la lista de coordenadas.
geometry = [Point(xy) for xy in coordenadas]
gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

# Representar el GeoDataFrame en un mapa
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
fig, ax = plt.subplots(figsize=(15, 10))
world.plot(ax=ax, color='lightgray')
gdf.plot(ax=ax, marker='o', color='red', markersize=50, alpha=0.7)
plt.title("Direcciones IP de origen y destino")
plt.show()