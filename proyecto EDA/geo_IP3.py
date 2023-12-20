
# -*- coding: utf-8 -*-
from geopy.geocoders import Nominatim
import time
import folium
import geopandas as gpd
import pandas as pd

# Cargar el conjunto de datos
df = pd.read_excel(r'E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\DATASETS\DATASETS MODIFICADOS\ciber_modificado_ciudades_final.xlsx')

# Función para geocodificar direcciones con Nominatim
def geocode_ips(direccion):
    geolocator = Nominatim(user_agent="my_geocoder")
    try:
        location = geolocator.geocode(direccion)
        if location:
            print(f"Coordenadas obtenidas para '{direccion}': {location.latitude}, {location.longitude}")
            return (location.latitude, location.longitude)
        else:
            print(f"No se pudieron obtener coordenadas para '{direccion}'")
            return None
    except Exception as e:
        print(f"Error geocodificando direccion '{direccion}': {e}")
    return None

# Verificar si la columna 'coordenadas' ya está presente en el DataFrame
if 'coordenadas' not in df.columns:
    df['coordenadas'] = df["Destino IP"].apply(geocode_ips)

# Contador de IPs procesadas
contador_total_ips = len(df)
contador_ips_con_coordenadas = 0
contador_ips_sin_coordenadas = 0

# Bucle para geocodificar las direcciones IP
for idx, row in df.iterrows():
    if contador_ips_con_coordenadas + contador_ips_sin_coordenadas == 1000:
        print("Se ha alcanzado el limite de 1000 IPs. Guardando resultados en HTML...")
        # Guardar el DF como un archivo HTML
        df.to_html(r"E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\coordenadas_ips_destino.html")
        break

    coordenadas = geocode_ips(row["Destino_IP"])
    if coordenadas:
        df.loc[idx, 'coordenadas'] = coordenadas
        contador_ips_con_coordenadas += 1
    else:
        contador_ips_sin_coordenadas += 1


# Imprimir resultados finales
print(f"\nNumero total de IPs procesadas: {contador_total_ips}")
print(f"Nnmero de IPs con coordenadas: {contador_ips_con_coordenadas}")
print(f"Numero de IPs sin coordenadas: {contador_ips_sin_coordenadas}")

print("Geolocalizacion completada. Resultados guardados en mapa_geolocalizacion.html.")
