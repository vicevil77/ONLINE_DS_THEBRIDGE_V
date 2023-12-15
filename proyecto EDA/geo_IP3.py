
import pandas as pd
import folium
from folium.plugins import PolyLineTextPath
from geoip2.database import Reader



# Ruta al archivo GeoLite2 City Database (descárgalo desde https://dev.maxmind.com/geoip/geoip2/geolite2/)
GEOIP_DATABASE_PATH = r"E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\BBDD geoloca\GeoLite2-City_20231212\GeoLite2-City.mmdb"  # Reemplaza con la ruta correcta

# Cargar el conjunto de datos
df = pd.read_excel(r'E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\DATASETS\DATASETS MODIFICADOS\ciber_modificado2.xlsx')

# Crear una lista vacía para guardar las coordenadas de las IPs
coords = []

# Crear el objeto Reader para acceder a la base de datos de GeoLite2 City
geoip_reader = Reader(GEOIP_DATABASE_PATH)

# Iterar sobre las filas del dataset y obtener las coordenadas de cada IP
for index, row in df.iterrows():
    ip_origen = row["Destino IP"]
    ip_destino = row["IP Origen"]

    try:
        # Obtener la información de geolocalización de la IP origen
        response_origen = geoip_reader.city(ip_origen)
        lat_origen, lon_origen = response_origen.location.latitude, response_origen.location.longitude

        # Obtener la información de geolocalización de la IP destino
        response_destino = geoip_reader.city(ip_destino)
        lat_destino, lon_destino = response_destino.location.latitude, response_destino.location.longitude

        coords.append([(lat_destino, lon_destino), (lat_origen, lon_origen)])

    except Exception as e:
        print(f"No se encontraron resultados para la IP: {ip_origen} o {ip_destino}. Error: {e}")

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
