# -*- coding: utf-8 -*-

import time
import folium
from folium.plugins import PolyLineTextPath
import geocoder
import pandas as pd

# Cargar el conjunto de datos
df = pd.read_excel(r'E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\DATASETS\DATASETS MODIFICADOS\ciber_modificado2.xlsx')

# Crear una lista vacía para guardar las coordenadas de las IPs
coords = []

# Definir la cantidad máxima de IPs a procesar en cada iteración (10,000 en este caso)
max_ips_per_iteration = 10000

# Iterar sobre las filas del dataset y obtener las coordenadas de cada IP
for index, row in df.iterrows():
    ip_origen = row["Destino IP"]
    ip_destino = row["IP Origen"]

    # Utilizar geocoder para obtener las coordenadas de la IP origen
    g_origen = geocoder.ip(ip_origen)

    # Utilizar geocoder para obtener las coordenadas de la IP destino
    g_destino = geocoder.ip(ip_destino)

    try:
        # Verificar que ambas respuestas tengan información de geolocalización
        if g_origen.latlng and g_destino.latlng:
            # Obtener las coordenadas de la IP origen y destino
            lat_origen, lon_origen = g_origen.latlng
            lat_destino, lon_destino = g_destino.latlng

            coords.append([(lat_destino, lon_destino), (lat_origen, lon_origen)])
    except Exception as e:
        print("No se encontraron resultados para la IP: {} o {}. Error: {}".format(ip_origen, ip_destino, e))
        # Continuar con la siguiente iteración
        continue

    # Verificar si se alcanzó el límite de IPs por iteración
    if index % max_ips_per_iteration == 0 and index > 0:
        # Guardar las coordenadas en un archivo
        df_coords = pd.DataFrame(coords, columns=["Coordenadas Destino", "Coordenadas Origen"])
        df_coords.to_csv("coordenadas_ips.csv", index=False)

        print("Procesadas {} IPs. El programa se detendra.".format(index).encode('utf-8').decode('utf-8'))
        break

    # Pausa de 30 segundos entre iteraciones
    time.sleep(30)

# Crear un mapa de Folium
m = folium.Map(location=[0, 0], zoom_start=2)

# Pintar las IPs con puntos usando la función Marker de Folium
for coord_pair in coords:
    folium.Marker(coord_pair[1], icon=folium.Icon(color='red'), popup='Destino IP').add_to(m)
    folium.Marker(coord_pair[0], icon=folium.Icon(color='blue'), popup='IP Origen').add_to(m)

# Unir las IPs origen con las de destino con líneas usando PolyLineTextPath
for coord_pair in coords:
    PolyLineTextPath(coord_pair, df["Tipo_ataque"]).add_to(m)

# Guardar el mapa como un archivo HTML
m.save(r"E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\mapa_ips.html")
