import requests
import pandas as pd
import folium
from folium.plugins import PolyLineTextPath

# Reemplaza 'TU_API_KEY' con tu propia clave de API de positionstack
API_KEY = '7acfe9588146c079e834a9742a62fce6'

# Cargar el conjunto de datos
df = pd.read_excel(r'E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\DATASETS\DATASETS MODIFICADOS\ciber_modificado2.xlsx')

# Crear una lista vacía para guardar las coordenadas de las IPs
coords = []

# Iterar sobre las filas del dataset y obtener las coordenadas de cada IP
for index, row in df.iterrows():
    # Obtener la IP origen y la IP destino
    ip_origen = row["IP Origen"]
    ip_destino = row["Destino IP"]

    # Hacer una petición a positionstack para obtener la información de geolocalización de la IP origen
    url_origen = f"http://api.positionstack.com/v1/forward?access_key={API_KEY}&query={ip_origen}"
    response_origen = requests.get(url_origen)
    data_origen = response_origen.json()

    if not data_origen["data"]:
        print(f"No se encontraron resultados para la IP origen: {ip_origen}")
        continue

    lat_origen = data_origen["data"][0]["latitude"]
    lon_origen = data_origen["data"][0]["longitude"]

    # Hacer lo mismo con la IP destino
    url_destino = f"http://api.positionstack.com/v1/forward?access_key={API_KEY}&query={ip_destino}"
    response_destino = requests.get(url_destino)
    data_destino = response_destino.json()

    if not data_destino["data"]:
        print(f"No se encontraron resultados para la IP destino: {ip_destino}")
        continue

    lat_destino = data_destino["data"][0]["latitude"]
    lon_destino = data_destino["data"][0]["longitude"]

    # Añadir las coordenadas de la IP origen y la IP destino a la lista coords
    coords.append([(lat_origen, lon_origen), (lat_destino, lon_destino)])

# Crear un mapa de Folium
m = folium.Map(location=[0, 0], zoom_start=2)

# Pintar las IPs con puntos usando la función Marker de Folium
for coord_pair in coords:
    folium.Marker(coord_pair[0], icon=folium.Icon(color='blue'), popup='IP Origen').add_to(m)
    folium.Marker(coord_pair[1], icon=folium.Icon(color='red'), popup='Destino IP').add_to(m)

# Unir las IPs origen con las de destino con líneas usando PolyLineTextPath
for coord_pair in coords:
    PolyLineTextPath(coord_pair, df["Tipo_ataque"]).add_to(m)

# Guardar el mapa como un archivo HTML
m.save(r"E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\mapa_ips.html")
