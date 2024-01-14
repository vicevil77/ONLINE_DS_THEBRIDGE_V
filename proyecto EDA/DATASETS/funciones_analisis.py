import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def pinta_distribucion_categoricas(df, columnas_categoricas, relativa=False, mostrar_valores=False):
    num_columnas = len(columnas_categoricas)
    num_filas = (num_columnas // 2) + (num_columnas % 2)

    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten() 

    for i, col in enumerate(columnas_categoricas):
        ax = axes[i]
        if relativa:
            total = df[col].value_counts().sum()
            serie = df[col].value_counts().apply(lambda x: x / total)
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis')
            ax.set_ylabel('Frecuencia Relativa')
        else:
            serie = df[col].value_counts()
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis')
            ax.set_ylabel('Frecuencia')

        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

        if mostrar_valores:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    for j in range(i + 1, num_filas * 2):
        axes[j].axis('off')

    plt.tight_layout()
    
    # Añadir leyenda
    if relativa:
        fig.legend(title='Legend Title', labels=['Frecuencia Relativa'], loc='upper right')
    
    plt.show()


def plot_categorical_relationship_fin(df, cat_col1, cat_col2, relative_freq=False, show_values=False, size_group = 5):
    # Prepara los datos
    count_data = df.groupby([cat_col1, cat_col2]).size().reset_index(name='count')
    total_counts = df[cat_col1].value_counts()
    
    # Convierte a frecuencias relativas si se solicita
    if relative_freq:
        count_data['count'] = count_data.apply(lambda x: x['count'] / total_counts[x[cat_col1]], axis=1)

    # Si hay más de size_group categorías en cat_col1, las divide en grupos de size_group
    unique_categories = df[cat_col1].unique()
    if len(unique_categories) > size_group:
        num_plots = int(np.ceil(len(unique_categories) / size_group))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * size_group:(i + 1) * size_group]
            data_subset = count_data[count_data[cat_col1].isin(categories_subset)]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=data_subset, order=categories_subset)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {cat_col1} y {cat_col2} - Grupo {i + 1}')
            plt.xlabel(cat_col1)
            plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de size_group categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=count_data)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {cat_col1} y {cat_col2}')
        plt.xlabel(cat_col1)
        plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()


def plot_categorical_numerical_relationship(df, categorical_col, numerical_col, show_values=False, measure='mean'):
    # Calcula la medida de tendencia central (mean o median)
    if measure == 'median':
        grouped_data = df.groupby(categorical_col)[numerical_col].median()
    else:
        # Por defecto, usa la media
        grouped_data = df.groupby(categorical_col)[numerical_col].mean()

    # Ordena los valores
    grouped_data = grouped_data.sort_values(ascending=False)

    # Si hay más de 5 categorías, las divide en grupos de 5
    if grouped_data.shape[0] > 5:
        unique_categories = grouped_data.index.unique()
        num_plots = int(np.ceil(len(unique_categories) / 5))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * 5:(i + 1) * 5]
            data_subset = grouped_data.loc[categories_subset]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=data_subset.index, y=data_subset.values)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {categorical_col} y {numerical_col} - Grupo {i + 1}')
            plt.xlabel(categorical_col)
            plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de 5 categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=grouped_data.index, y=grouped_data.values)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {categorical_col} y {numerical_col}')
        plt.xlabel(categorical_col)
        plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()


def plot_combined_graphs(df, columns, whisker_width=1.5, bins = None): #NUMERICAS
    num_cols = len(columns)
    if num_cols:
        
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, 5 * num_cols))
        print(axes.shape)

        for i, column in enumerate(columns):
            if df[column].dtype in ['int64', 'float64']:
                # Histograma y KDE
                sns.histplot(df[column], kde=True, ax=axes[i,0] if num_cols > 1 else axes[0], bins= "auto" if not bins else bins)
                if num_cols > 1:
                    axes[i,0].set_title(f'Histograma y KDE de {column}')
                else:
                    axes[0].set_title(f'Histograma y KDE de {column}')

                # Boxplot
                sns.boxplot(x=df[column], ax=axes[i,1] if num_cols > 1 else axes[1], whis=whisker_width)
                if num_cols > 1:
                    axes[i,1].set_title(f'Boxplot de {column}')
                else:
                    axes[1].set_title(f'Boxplot de {column}')

        plt.tight_layout()
        plt.show()

def plot_grouped_boxplots(df, cat_col, num_col):#NUMERICAS
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)
    group_size = 5

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cat_col, y=num_col, data=subset_df)
        plt.title(f'Boxplots of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xticks(rotation=45)
        plt.show()



def plot_grouped_histograms(df, cat_col, num_col, group_size):
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        for cat in subset_cats:
            sns.histplot(subset_df[subset_df[cat_col] == cat][num_col], kde=True, label=str(cat))
        
        plt.title(f'Histograms of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xlabel(num_col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()



def grafico_dispersion_con_correlacion(df, columna_x, columna_y, tamano_puntos=50, mostrar_correlacion=False):
    """
    Crea un diagrama de dispersión entre dos columnas y opcionalmente muestra la correlación.

    Args:
    df (pandas.DataFrame): DataFrame que contiene los datos.
    columna_x (str): Nombre de la columna para el eje X.
    columna_y (str): Nombre de la columna para el eje Y.
    tamano_puntos (int, opcional): Tamaño de los puntos en el gráfico. Por defecto es 50.
    mostrar_correlacion (bool, opcional): Si es True, muestra la correlación en el gráfico. Por defecto es False.
    """

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=columna_x, y=columna_y, s=tamano_puntos)

    if mostrar_correlacion:
        correlacion = df[[columna_x, columna_y]].corr().iloc[0, 1]
        plt.title(f'Diagrama de Dispersión con Correlación: {correlacion:.2f}')
    else:
        plt.title('Diagrama de Dispersión')

    plt.xlabel(columna_x)
    plt.ylabel(columna_y)
    plt.grid(True)
    plt.show()

    
def graficos_dispersión_combinados(datos, categorias, colores=None):
    """
    Combina múltiples gráficos de dispersión bivariantes en uno solo.

    Parámetros:
    - datos: Un diccionario donde las claves son las categorías y los valores son matrices 2D con los datos.
    - categorias: Lista de categorías en el orden en que deben aparecer en el gráfico combinado.
    - colores: Lista opcional de colores para cada categoría.

    """
    if colores is None:
        colores = plt.cm.jet(np.linspace(0, 1, len(categorias)))

    fig, ax = plt.subplots()

    for categoria, color in zip(categorias, colores):
        datos_categoria = datos[categoria]
        x = datos_categoria[:, 0]
        y = datos_categoria[:, 1]
        ax.scatter(x, y, label=categoria, color=color)

    ax.set_xlabel('Variable X')
    ax.set_ylabel('Variable Y')
    ax.legend()
    plt.show()

    # Supongamos que tienes un diccionario llamado datos con claves 'Categoria1', 'Categoria2', ...
# y valores que son matrices 2D de datos.

# datos = {'Categoria1': np.random.rand(50, 2),
#          'Categoria2': np.random.rand(50, 2),
#          'Categoria3': np.random.rand(50, 2)}

# categorias = ['Categoria1', 'Categoria2', 'Categoria3']

# graficos_dispersión_combinados(datos, categorias)



def bubble_plot(df, col_x, col_y, col_size, scale = 1000):
    """
    Crea un scatter plot usando dos columnas para los ejes X e Y,
    y una tercera columna para determinar el tamaño de los puntos.

    Args:
    df (pd.DataFrame): DataFrame de pandas.
    col_x (str): Nombre de la columna para el eje X.
    col_y (str): Nombre de la columna para el eje Y.
    col_size (str): Nombre de la columna para determinar el tamaño de los puntos.
    """

    # Asegúrate de que los valores de tamaño sean positivos
    sizes = (df[col_size] - df[col_size].min() + 1)/scale

    plt.scatter(df[col_x], df[col_y], s=sizes)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f'Burbujas de {col_x} vs {col_y} con Tamaño basado en {col_size}')
    plt.show()

def columnas_cardin(df):
# Obtener todas las columnas
    todas_las_columnas = df.columns

    # Imprimir encabezados
    print(f"{'Columna': <15}{'Cardinalidad': <15}{'Tipo': <20}")

    for col in todas_las_columnas:
        if col in df.select_dtypes(include=[np.number]).columns:
            # Columnas numéricas
            cardinalidad = df[col].nunique()

            if cardinalidad == 2:
                tipo_columna = 'binaria'
            elif cardinalidad == 1:
                tipo_columna = 'constante'
            elif df[col].dtype == np.int64:
                tipo_columna = 'numérica discreta'
            elif df[col].dtype == np.float64:
                tipo_columna = 'numérica continua'
            else:
                tipo_columna = 'otro'

        else:
            # Columnas no numéricas
            cardinalidad = df[col].nunique()

            # Identificar columnas binarias
            if cardinalidad == 2:
                tipo_columna = 'binaria'
            else:
                # Intentar determinar si es categórica ordinal o nominal
                try:
                    is_ordinal = df[col].apply(lambda x: pd.api.types.CategoricalDtype.ordered if pd.notna(x) else np.nan).dropna().unique()[0]
                    # la línea lambda se utiliza para determinar si una columna categórica es ordenada. Primero, convierte la columna a un tipo categórico ordenado 
                    # si no es un valor nulo, y luego, después de eliminar los nulos y obtener los valores únicos, devuelve el primer valor único. 
                    # Esto se hace para verificar si todos los valores únicos son nulos (NaN), lo que indica que la columna original no era ordenada.
                    if is_ordinal:
                        tipo_columna = 'categórica ordinal'
                    else:
                        tipo_columna = 'categórica nominal'
                except Exception as e:
                    tipo_columna = 'otro'

        # Imprimir la información de cada columna
        print(f"{col: <15}{cardinalidad: <15}{tipo_columna: <20}")


def analizar_dataframe(df):
    # Crear un DataFrame de resumen
    resumen = pd.DataFrame(index=df.columns)
    
    # Identificar el tipo de cada columna
    resumen['Tipo'] = df.dtypes
    
    # Contar la cantidad de valores únicos y calcular la cardinalidad
    resumen['Cardinalidad'] = df.nunique()
    
    # Calcular el porcentaje de valores nulos en cada columna
    resumen['Porcentaje_Nulos'] = (df.isnull().mean()) * 100
    
    # Clasificar las columnas en categóricas, numéricas y binarias
    resumen['Clasificación'] = 'Desconocida'
    
    # Identificar columnas categóricas
    categóricas = df.select_dtypes(include='object').columns
    resumen.loc[categóricas, 'Clasificación'] = 'Categórica'
    
    # Identificar columnas numéricas
    numéricas = df.select_dtypes(include=['int', 'float']).columns
    resumen.loc[numéricas, 'Clasificación'] = 'Numérica'
    
    # Identificar columnas binarias (que tienen solo dos valores únicos)
    binarias = df.columns[df.nunique() == 2]
    resumen.loc[binarias, 'Clasificación'] = 'Binaria'
    
    return resumen


def convertir_notacion_india_numero_occidental(cantidad_india):
    # Verificar si la entrada ya es un número
    if isinstance(cantidad_india, (int, float)):
        return cantidad_india  # No es una cadena, devolver el valor original

    partes = cantidad_india.split(',')

    crore = 0
    lakh = 0
    unidades = 0

    if len(partes) == 1:
        unidades = int(partes[0])
    elif len(partes) == 2:
        lakh, unidades = map(int, partes)
    elif len(partes) == 3:
        crore, lakh, unidades = map(int, partes)

    numero_occidental = (crore * 10000000) + (lakh * 100000) + unidades

    return numero_occidental


#PARA HACER DIAGRAMAS DE FLUJO
from graphviz import Digraph
import pandas as pd

# Abrir el DataFrame
archivos_xlsx = r'E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\DATASETS\DATASETS MODIFICADOS\ciber_modificado.xlsx'
df_geo = pd.read_excel(archivos_xlsx, engine='openpyxl')

# Crear un grafo dirigido (diagrama de flujo)
G = Digraph()

# Agregar nodos al grafo para cada columna en el DataFrame
for col in df_geo.columns:
    G.node(col)

# Agregar arcos (flechas) entre nodos para representar el flujo de datos
for i in range(len(df_geo.columns) - 1):
    G.edge(df_geo.columns[i], df_geo.columns[i + 1])

# Guardar el grafo como un archivo de imagen (por ejemplo, en formato PNG)
imagen_path = r'E:\Cursos\BC_Data_Science\Repositorio\ONLINE_DS_THEBRIDGE_V\proyecto EDA\DIAGRAMAS DE FLUJO\imagen.png'
G.render(imagen_path, format='png', engine='dot', cleanup=True)

print(f"Diagrama de flujo guardado en: {imagen_path}")



# geolocalizar con google maps(NO FUNCIONO)

from googletrans import Translator
def traducir_columna(df, columna, idioma_destino):
    # Definir la función para traducir si es necesario
    def traducir(texto):
        try:
            # Convertir el texto a cadena si no es de tipo str
            texto = str(texto)

            # Verificar si el texto contiene solo números
            if texto.isdigit():
                return texto  # No traducir números, devolver el texto original

            # Traducir solo si el idioma detectado no es el idioma de destino
            idioma_detectado = detect(texto)
            if idioma_detectado != idioma_destino:
                # Intentar la traducción y manejar el error específico
                try:
                    traduccion = translate(texto, idioma_destino)
                    if traduccion:
                        return traduccion
                    else:
                        print(f"La traducción para el texto '{texto}' es None.")
                        return texto
                except Exception as e:
                    print(f"Error durante la traducción: {e}. No se puede traducir el texto.")
                    return texto
            else:
                return texto

        except Exception as e:
            print(f"Error durante la traducción: {e}")
            return texto

    # Aplicar la función de traducción a la columna
    df[columna + '_traducido'] = df[columna].apply(traducir)
    return df

# Ejemplo de uso
#df = df_esp
#columna = "Payload Data"
#idioma_destino = 'es'
#df_traducido = traducir_columna(df, columna, idioma_destino)



# obetenr todo de un dataset INFORMACION GENERAL:
def obtener_estadisticas(df):

    resultado = pd.DataFrame()
    for col in df.columns:
        datos = {}
        datos['porcentaje_cardinalidad'] = round(df[col].nunique() / len(df) * 100, 2)
        datos['Tipo'] = df[col].dtype
        if pd.api.types.is_numeric_dtype(df[col]):
            datos['media'] = round(df[col].mean(), 2)
            datos['moda'] = "No"
            datos['std'] = round(df[col].std(), 2)
            datos['var'] = round(df[col].var(), 2)
            datos['Q1'] = round(df[col].quantile(0.25), 2)
            datos['mediana'] = round(df[col].median(), 2)
            datos['Q3'] = round(df[col].quantile(0.75), 2)
            datos['Categoria'] = 'numerica continua' if df[col].nunique() > 10 else 'numerica discreta'
        else:
            datos['media'] = "No"
            datos['moda'] = df[col].mode().iloc[0] if not df[col].mode().empty else "No"
            datos['std'] = "No"
            datos['var'] = "No"
            datos['Q1'] = "No"
            datos['mediana'] = "No"
            datos['Q3'] = "No"
            datos['Categoria'] = 'categorica ordinal' if df[col].nunique() > 2 else 'categorica nominal'
        datos['porcentaje_NaN'] = round(df[col].isna().mean() * 100, 2)
        resultado[col] = pd.Series(datos)
    return resultado.transpose()