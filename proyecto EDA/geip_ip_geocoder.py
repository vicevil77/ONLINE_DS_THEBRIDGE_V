# -*- coding: utf-8 -*-
import ipwhois
import time
import folium
from folium.plugins import PolyLineTextPath
import pandas as pd
import requests


# Cargar el conjunto de datos
df = pd.read_excel(r'E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\DATASETS\DATASETS MODIFICADOS\ciber_modificado2.xlsx')

# Define tu clave de API de ipstack
api_key = '5dbd572c7b96c5029e04f62e98d19e8f'

# Crear una lista vacía para guardar las coordenadas de las IPs
coords = []

# Definir la cantidad máxima de IPs a procesar en cada iteración (10,000 en este caso)
max_ips_per_iteration = 1000

# Iterar sobre las filas del dataset y obtener las coordenadas de cada IP
for index, row in df.iterrows():
    ip_origen = row["Destino IP"]
    ip_destino = row["IP Origen"]

    try:
        # Utilizar la API de ipstack para obtener información de geolocalización
        api_url_destino = f"http://api.ipstack.com/{ip_destino}?access_key={api_key}"
        response_destino = requests.get(api_url_destino)
        result_destino = response_destino.json()

        api_url_origen = f"http://api.ipstack.com/{ip_origen}?access_key={api_key}"
        response_origen = requests.get(api_url_origen)
        result_origen = response_origen.json()

        # Extraer las coordenadas de latitud y longitud
        lat_destino = result_destino.get('latitude')
        lon_destino = result_destino.get('longitude')

        lat_origen = result_origen.get('latitude')
        lon_origen = result_origen.get('longitude')

        if lat_destino and lon_destino and lat_origen and lon_origen:
            coords.append([(float(lat_destino), float(lon_destino)), (float(lat_origen), float(lon_origen))])

    except Exception as e:
        print(f"No se encontraron resultados para la IP: {ip_origen} o {ip_destino}. Error: {e}")
        continue

    # Verificar si se alcanzó el límite de IPs por iteración
    if index % max_ips_per_iteration == 0 and index > 0:
        # Crear un mapa de Folium
        m = folium.Map(location=[0, 0], zoom_start=2)

        # Pintar las IPs con puntos usando la función Marker de Folium
        for coord_pair in coords:
            folium.Marker(coord_pair[1], icon=folium.Icon(color='red'), popup='Destino IP').add_to(m)
            folium.Marker(coord_pair[0], icon=folium.Icon(color='blue'), popup='IP Origen').add_to(m)

        # Unir las IPs origen con las de destino con líneas usando PolyLine
        for coord_pair in coords:
            folium.PolyLine([coord_pair[1], coord_pair[0]], color='green').add_to(m)

        # Guardar el mapa como un archivo HTML
        m.save(r"E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\mapa_ips.html")

        print("Procesadas {} IPs. El programa se detendra.".format(index))
        break

    # Pausa de 90 segundos entre iteraciones
    time.sleep(90)
        

