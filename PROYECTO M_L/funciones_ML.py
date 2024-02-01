
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_percentage_error


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


def plot_combined_graphs(df, columns, whisker_width=1.5, bins = None):
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

def plot_grouped_boxplots(df, cat_col, num_col):
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


def get_features_num_regression(df, target_col,umbral_corr = 0.5, pvalue = None):
    """
    Devuelve una lista con las columnas numéricas del dataframe cuya correlación con la columna designada por "target_col" sea superior en valor absoluto al valor dado por "umbral_corr".

    Args:
        df: El DataFrame del que se quiere obtener las características correlacionadas.
        target_col: El nombre de la columna objetivo.
        umbral_corr= cantidad numerico entre 0 a 1
    
    Returns:
        Una lista con las columnas correlacionadas.
    """
    # comprbamos en una lista de comprensión si todas son columnas numericas (true o false)
    columnas_num = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    #verificacion de columnas no numericas, describiendo el tipo y descartamos éstas
    columnas_no_num= set(df.columns) - set(columnas_num)
    if columnas_no_num:
        print("Las columnas no numéricas descartadas en esta función son:", columnas_no_num)
        df= df[columnas_num]
    
    # Calculamos las correlaciones y p-values utilizando f_regression, sin la target, obteniendo la X en la funcion de regresión .
    _, p_values = f_regression(df.drop(columns=[target_col]), df[target_col])

    # Filtramos las columnas según el umbral_corr y p-value, incluyendo la target nuevamente en la lista de correlaciones
    columnas_correlacionadas_dir = df.columns[(df.corr()[target_col] >= umbral_corr)].tolist()
    columnas_correlacionadas_indir = df.columns[(df.corr()[target_col] >= -(umbral_corr))].tolist()

    # Filtramos por significancia si se proporciona un p-value, agrupando columnas con su p_values
    if pvalue is not None:
        columnas_correlacionadas_dir = [col for col, p_val in zip(columnas_correlacionadas_dir, p_values) if p_val <= pvalue]
        columnas_correlacionadas_indir = [col for col, p_val in zip(columnas_correlacionadas_indir, p_values) if p_val <= pvalue]

    nl ='\n'
    print(f"Las columnas numéricas con |valor de correlación superior| a {umbral_corr} aportado en la variable 'umbral_corr' en referencia a {target_col} son: {nl}Directamente proporcionales:   {columnas_correlacionadas_dir} {nl}Indirectamente proporcionales: {columnas_correlacionadas_indir}")



def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Crea un conjunto de pair plots para visualizar las correlaciones entre las columnas numéricas del DataFrame.

    Args:
        df: El DataFrame del que se quiere visualizar las correlaciones.
        target_col: El nombre de la columna objetivo.
        umbral_corr= numbral maximo establecido de correlacion con la target
        pvalue: El valor de p-valor.

    Returns:
        None
    """

    columnas_para_pintar = []
    columnas_umbral_mayor = []

    #iteramos por la columnas
    for col in columns:
        #si en la iteracion de las columnas del DF y siempre que...
        # se comprube si son numéricas(true) o no son numéricas(false)
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            # usando el indice de correlación de Pearson y el p-valor(funcion pearsonr)
            # calculamos dichos parametros para target y resto de columnas
            corr, pv = pearsonr(df[col], df[target_col])
            if abs(corr) > umbral_corr:
                columnas_umbral_mayor.append(col)
                if pvalue is None or pv < pvalue:
                    columnas_para_pintar.append(col)

    # Número máximo de gráficas por grupo
    max_graficas_por_grupo = 6

    # Dividir en grupos según el número máximo de gráficas
    len(columnas_para_pintar) // max_graficas_por_grupo
    # En una lista de comprension, iteramos en rango desde 0 hasta el numero de columnas a pintar, por cada grupo maximo establecido
    # creando graficas con columnas maxi de i+ grupo max establecido ( ejem: '0 hasta 0+6)
    columnas = [columnas_para_pintar[i:i+max_graficas_por_grupo] for i in range(0, len(columnas_para_pintar), max_graficas_por_grupo)]

    # iteramos por i y por valor 'umbral_corr' establecido a cada grupo en cada iteración,  creeando pair plots para cada grupo,
    for i, grupo in enumerate(columnas):
        sns.pairplot(df, vars=grupo, hue=target_col)
        plt.suptitle(f"Group {i}", y=1.02)# creo nombres de grupo un poco por encima de y, para que no se superponga con la gráfica
        plt.show()
    
    return "Las columnas con un mayor umbral_corr al establecido son", columnas_umbral_mayor





def get_features_cat_regression(dataframe: pd.DataFrame, target_col: str, pvalue: float = 0.05) -> list:
    """
    Esta función recibe un dataframe y dos argumentos adicionales: 'target_col' y 'pvalue'.
    
    Parámetros:
    - dataframe: DataFrame de pandas.
    - target_col: Nombre de la columna que actuará como el objetivo para un modelo de regresión.
    - pvalue: Valor de p umbral para la significancia estadística (por defecto es 0.05).
    
    Devuelve:
    - Una lista con las columnas categóricas cuya relación con 'target_col' es estadísticamente significativa.
    - None si hay errores en los parámetros de entrada.
    """
    # Comprueba si 'target_col' es una columna numérica válida en el dataframe
    if target_col not in dataframe.columns or not pd.api.types.is_numeric_dtype(dataframe[target_col]):
        print(f"Error: '{target_col}' no es una columna numérica válida en el dataframe.")
        return None
    
    # Comprueba si 'pvalue' es un float válido
    if not isinstance(pvalue, float):
        print("Error: 'pvalue' debería ser un float.")
        return None
    
    # Identifica las columnas categóricas
    cat_columns = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Comprueba si hay columnas categóricas
    if not cat_columns:
        print("Error: No se encontraron columnas categóricas en el dataframe.")
        return None
    
    # Realiza pruebas estadísticas y filtra columnas basadas en el valor de p
    selected_columns = []
    for cat_col in cat_columns:
        contingency_table = pd.crosstab(dataframe[cat_col], dataframe[target_col])
        
        # Elige la prueba apropiada según la cardinalidad
        if len(contingency_table) > 2:
            _, p, _, _ = chi2_contingency(contingency_table)
        else:
            _, p = f_oneway(*[dataframe[target_col][dataframe[cat_col] == category] for category in dataframe[cat_col].unique()])
        
        if p < pvalue:
            selected_columns.append(cat_col)
    
    return selected_columns





def plot_features_cat_regression(dataframe: pd.DataFrame, target_col: str = "", 
                                  columns: list = [], pvalue: float = 0.05, 
                                  with_individual_plot: bool = False) -> list:
    """
    Esta función recibe un dataframe y varios argumentos opcionales para visualizar y analizar la relación
    entre variables categóricas y una columna objetivo en un modelo de regresión.

    Parámetros:
    - dataframe: DataFrame de pandas.
    - target_col: Nombre de la columna que actuará como el objetivo para un modelo de regresión.
    - columns: Lista de nombres de columnas categóricas a considerar (por defecto, todas las numéricas).
    - pvalue: Valor de p umbral para la significancia estadística (por defecto es 0.05).
    - with_individual_plot: Booleano que indica si se deben incluir gráficos individuales para cada columna (por defecto es False).

    Devuelve:
    - Una lista con las columnas seleccionadas que cumplen con las condiciones de significancia.
    - None si hay errores en los parámetros de entrada.
    """
    # Comprueba si 'target_col' es una columna numérica válida en el dataframe
    if target_col and (target_col not in dataframe.columns or not pd.api.types.is_numeric_dtype(dataframe[target_col])):
        print(f"Error: '{target_col}' no es una columna numérica válida en el dataframe.")
        return None
    
    # Comprueba si 'pvalue' es un float válido
    if not isinstance(pvalue, float):
        print("Error: 'pvalue' debería ser un float.")
        return None
    
    # Comprueba si 'columns' es una lista válida de strings
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        print("Error: 'columns' debería ser una lista de strings.")
        return None
    
    # Comprueba si 'with_individual_plot' es un booleano válido
    if not isinstance(with_individual_plot, bool):
        print("Error: 'with_individual_plot' debería ser un booleano.")
        return None
    
    # Si 'columns' está vacío, utiliza todas las columnas numéricas en el dataframe
    if not columns:
        columns = dataframe.select_dtypes(include=['number']).columns.tolist()
    
    # Filtra columnas basadas en pruebas estadísticas
    selected_columns = get_features_cat_regression(dataframe, target_col, pvalue)
    selected_columns = list(set(selected_columns) & set(columns))
    
    if not selected_columns:
        print("Ninguna columna cumple con las condiciones especificadas para trazar.")
        return None
    
    # Histogramas
    for cat_col in selected_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=dataframe, x=target_col, hue=cat_col, multiple="stack", kde=True)
        plt.title(f"Histograma para {target_col} por {cat_col}")
        plt.show()
    
    return selected_columns





def mejores_hiperparametros_regresion_lineal(train_X, train_y, test_X, test_y):
    """funcion para que te diga los mejores hiperparametros para ajustar la generalziacion de un modelo de regresion lienal: simple, multipleo generalizada
    Se puede usar conmodelos de clasificacion pero con la target categorica

    arg: (train_X, train_y, test_X, test_y
     
    """
       # Definir el modelo
    model = ElasticNet()

    # Definir los parámetros a buscar
    params = {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],  # Valores de alpha para la regularización
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Proporción entre L1 y L2 en regresión elástica
    }

    # Definir la métrica a optimizar (en este caso, MAE)
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Realizar la búsqueda en cuadrícula con validación cruzada
    grid_search = GridSearchCV(model, params, scoring=scorer, cv=5)
    grid_search.fit(train_X, train_y)

    # Mostrar los mejores hiperparámetros encontrados
    best_params = grid_search.best_params_
    print("Mejores hiperparámetros:", best_params)

    # Entrenar el modelo con los mejores hiperparámetros
    best_model = grid_search.best_estimator_
    best_model.fit(train_X, train_y)

    # Hacer predicciones en el conjunto de prueba
    predictions = best_model.predict(test_X)

    # Calcular métricas en el conjunto de prueba
    mae_test = mean_absolute_error(test_y, predictions)
    mse_test = mean_squared_error(test_y, predictions)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(test_y, predictions)
    mape= mean_absolute_percentage_error(test_y, predictions)

    # Mostrar métricas en el conjunto de prueba
    print("MAE test:", mae_test)
    print("MSE test:", mse_test)
    print("RMSE test:", rmse_test)
    print("R2 test:", r2_test)
    print("MAPE test:", mape)


#funcion para redimensionalr variables a 2 dimensiones  que puedan ser usadas es un arbol de decision
def reshape_test_data(test_X):
    """Redimensiona los datos de prueba a una matriz bidimensional.

    Args:
        test_X: Los datos de prueba a redimensionar.

    Returns:
        Los datos de prueba redimensionados.
    """
    if isinstance(test_X, pd.Series):
        # Si test_X es una Series de pandas, conviértela a un array de NumPy
        test_X_array = test_X.to_numpy()
    elif isinstance(test_X, pd.DataFrame):
        # Si test_X es un DataFrame de pandas, conviértelo a un array de NumPy
        test_X_array = test_X.to_numpy()
    else:
        # Si test_X ya es un array de NumPy, no es necesario convertirlo
        test_X_array = test_X

    # Utiliza reshape si es necesario
    if len(test_X_array.shape) == 1:
        n_samples = test_X_array.shape[0]
        return test_X_array.reshape(n_samples, -1)
    else:
        return test_X_array

#funcion para sacar MAE sin sklearn
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

#funcion paea sacar el MAPE sin skleran
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100