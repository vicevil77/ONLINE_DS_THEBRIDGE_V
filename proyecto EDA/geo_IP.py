import pandas as pd
import requests
import json
import time  # Importar el módulo time para agregar retraso entre las solicitudes
import re
import chardet

# abrimos el dataframe
archivos_xlsx = r'E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\DATASETS\DATASETS MODIFICADOS\ciber_modificado.xlsx'
df= pd.read_excel(archivos_xlsx, engine='openpyxl')

# Eliminar filas con valores NaN en las columnas "IP Origen" y "IP Destino"
df_filtrado = df.dropna(subset=["IP Origen", "Destino IP"])


# Convertir las columnas de IP a números
df["IP Origen"] = df["IP Origen"].astype(int)
df["Destino IP"] = df["Destino IP"].astype(int)

#comporbar si la cadena es valida
def es_dirección_ip_válida(ip):
    if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip):
        raise ValueError("Invalid IP address: {}".format(ip))

    # Reemplace el byte 0xf3 con un byte UTF-8 válido
    cadena = "es_dirección_ip_válida"
    cadena = cadena.replace(chr(0xf3), chr(0xe1))

    # Compruebe si la cadena ahora está codificada en UTF-8
    with open("geo_IP.py", "rb") as f:
        encoding = chardet.detect(f.read())

        if encoding["encoding"] != "utf-8":
            print("La cadena '{}' aún no está codificada en UTF-8. Por favor, corrija el error antes de ejecutar el script.".format(cadena))
            exit(1)
        
# Leer los datos
df = pd.read_excel("E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\DATASETS\DATASETS MODIFICADOS\ciber_modificado.xlsx")

# Filtrar por direcciones IP válidas
df_filtrado = df[df["IP Origen"].apply(es_dirección_ip_válida)]


# Tu clave de API de Google Maps
with open(r"E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\api_key.txt", "r") as f:
    API_KEY = f.read()
    
def get_geolocation(ip):
    if not es_dirección_ip_válida(ip):
        return None

    response = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json?address={}".format(ip)
    )
    data = json.loads(response.content)
    return data["results"][0]["geometry"]["location"]

# Obtener datos de geolocalización para direcciones IP válidas
geo_df = pd.DataFrame(
    [get_geolocation(ip) for ip in df_filtrado["IP Origen"]],
    columns=["lat", "lon"],)
   

# Crear el mapa
import folium

mapa = folium.Map(location=[0, 0], zoom_start=2)

# Añadir marcadores al mapa
for i in range(len(df)):
    lat = geo_df.loc[i, "lat"]
    lon = geo_df.loc[i, "lon"]
    tipo_ataque = df.loc[i, "Tipo_ataque"]
    color = "red" if tipo_ataque == "Malware" else "blue"
    folium.Marker([lat, lon], popup=tipo_ataque, icon=folium.Icon(color=color)).add_to(mapa)

# Agregar información al mapa
for i in range(len(df)):
    lat = geo_df.loc[i, "lat"]
    lon = geo_df.loc[i, "lon"]
    ciudad_origen = df.loc[i, "Datos_geolocalizacion"].split(",")[0]
    ciudad_destino = df.loc[i, "Datos_geolocalizacion"].split(",")[1]
    folium.Marker([lat, lon], popup="Origen: {} - Destino: {}".format(ciudad_origen, ciudad_destino)).add_to(mapa)

# Guardar el mapa
mapa.save(r'E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\mapa_interactivo.html')
