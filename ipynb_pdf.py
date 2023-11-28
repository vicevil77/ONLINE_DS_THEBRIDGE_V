import os
import subprocess

def convertir_ipynb_en_carpeta_a_pdf(ruta_carpeta):
    # Obtén la lista de archivos .ipynb en la carpeta
    archivos_ipynb = [archivo for archivo in os.listdir(ruta_carpeta) if archivo.endswith('.ipynb')]

    # Crea una carpeta para almacenar los archivos PDF si no existe
    carpeta_salida_pdf = os.path.join(ruta_carpeta, 'pdf_output')
    os.makedirs(carpeta_salida_pdf, exist_ok=True)

    # Convierte cada archivo .ipynb a PDF
    for archivo_ipynb in archivos_ipynb:
        ruta_notebook = os.path.join(ruta_carpeta, archivo_ipynb)
        nombre_pdf = os.path.splitext(archivo_ipynb)[0] + '.pdf'
        ruta_salida_pdf = os.path.join(carpeta_salida_pdf, nombre_pdf)

        # Construye el comando para nbconvert
        comando = f'jupyter nbconvert --to pdf "{ruta_notebook}" --output "{nombre_pdf}"'

        # Imprime el comando para depuración
        print(f'Ejecutando comando: {comando}')

        # Ejecuta el comando
        subprocess.run(comando, shell=True)

if __name__ == "__main__":
    # Especifica la ruta de la carpeta que contiene los notebooks
    carpeta_de_trabajo = r"E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\SPRING 4\UNIT 2"
    convertir_ipynb_en_carpeta_a_pdf(carpeta_de_trabajo)