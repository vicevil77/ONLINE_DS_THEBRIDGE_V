
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_regression
from sklearn.metrics import make_scorer, mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import pearsonr, chi2_contingency, chi2, f_oneway
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import ptitprince as pt


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


def convertir_object_a_numeros(df):
    for columna in df.columns:
        if df[columna].dtype == 'object':  
            categorias = df[columna].unique() 
            mapeo = {categoria: valor for valor, categoria in enumerate(categorias)} 
            df[columna] = df[columna].map(mapeo)  
    return df


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)


def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    


def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = dbscan.core_sample_indices_
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~np.isin(np.arange(len(X)), np.concatenate([core_mask, np.where(anomalies_mask)[0]]))

    plt.scatter(X.loc[core_mask, X.columns[0]], X.loc[core_mask, X.columns[1]], c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(X.loc[core_mask, X.columns[0]], X.loc[core_mask, X.columns[1]], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(X.loc[anomalies_mask, X.columns[0]], X.loc[anomalies_mask, X.columns[1]], c="r", marker="x", s=100)
    plt.scatter(X.loc[non_core_mask, X.columns[0]], X.loc[non_core_mask, X.columns[1]], c=dbscan.labels_[non_core_mask], marker=".")

    plt.xlabel("$x_1$", fontsize=14) if show_xlabels else plt.tick_params(labelbottom=False)
    plt.ylabel("$x_2$", fontsize=14, rotation=0) if show_ylabels else plt.tick_params(labelleft=False)
    
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)

   


def plot_clusterer_comparacion(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='b'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=15, linewidths=20,
                color=cross_color, zorder=11, alpha=1)
    

def best_eps(df, min_eps, max_eps, paso_eps):
    mejor_eps = None
    mejor_silueta = -1
    
    for eps in range(min_eps, max_eps, paso_eps):
        dbscan = DBSCAN(eps=eps)
        labels = dbscan.fit_predict(df)
        
        # Ignorar el caso donde solo hay un grupo (no se puede calcular la silueta)
        if len(set(labels)) > 1:
            silueta = silhouette_score(df, labels)
            if silueta > mejor_silueta:
                mejor_silueta = silueta
                mejor_eps = eps
    
    return mejor_eps, mejor_silueta



def encontrar_n_clusters(df, max_clusters=20):
    
    X = df.values

    # variables
    inercias = []
    silhouette_scores = []
    
    # Calcular inercia y puntuación de silueta para diferentes clusters
    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)# encuentro los cluesteres
        
        # Calcular la inercia y la puntuación de silueta
        inercias.append(kmeans.inertia_)
        if i > 1:
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    # Dibujar el gráfico del método del codo
    plt.figure(figsize=(20, 6))
    plt.subplot(121)
    plt.plot(range(1, max_clusters+1), inercias, marker='o')
    plt.title('Método del Codo')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    
    # Dibujar el gráfico de la puntuación de silueta
    plt.subplot(122)
    plt.plot(range(2, max_clusters+1), silhouette_scores, marker='o')
    plt.title('Método de la Silueta')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Puntuación de Silueta')
    plt.show();


def plot_silueta_score_con_K(df, max_k):
    # almacenar los valores
    silhouette_scores = []

    # Rango de valores de k que deseas probar
    k_values = range(2, max_k + 1)

    # Iterar sobre diferentes valores de k
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        find_labels_clusters = kmeans.fit_predict(df)
        
        # Calcular la puntuación de silueta para el clustering actual
        silueta_point = silhouette_score(df, find_labels_clusters)
        silhouette_scores.append(silueta_point)

    # Trazar el gráfico de la evolución del score de la silueta
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Puntuación de Silueta')
    plt.title('Evolución del "Score de Silueta" para Diferentes Valores de k')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show();


def plot_grafica_variance_ratio_explain(df, figsize=(10, 5)):
    """
    Grafica la varianza explicada por cada componente principal en un análisis de PCA.

    Args:
        df (pandas.DataFrame): El DataFrame de entrada que contiene los datos.
        figsize (tuple, optional): El tamaño de la figura. Defaults to (10, 5).

    Returns:
        matplotlib.axes._axes.Axes: El gráfico generado.
    """

    # Realizamos PCA con un máximo de componentes igual al número de muestras
    pca = PCA(n_components=min(df.shape[0], df.shape[1]))
    pca.fit(df)

    # Creamos el gráfico usando Seaborn
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        x=list(range(1, pca.n_components_ + 1)),
        y=pca.explained_variance_ratio_,
        color='skyblue',
        ax=ax
    )

    # Anotamos cada barra con la varianza explicada
    for i, y in enumerate(pca.explained_variance_ratio_):
        ax.text(i, y + 0.02, f"{y:.2f}", ha='center', fontsize=12)

    # Personalizamos las etiquetas y el título del eje
    ax.set_xlabel('Componente Principal', fontsize=14)
    ax.set_ylabel('Varianza Explicada', fontsize=14)
    ax.set_title(f'Varianza Explicada por Componente Principal (máximo {pca.n_components_} componentes)')

    # Ajustamos los límites del eje y la cuadrícula
    ax.set_xticks(range(1, pca.n_components_ + 1))
    ax.set_xlim(0.5, pca.n_components_ + 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    return ax


def graficar_VE(ve, n_componentes, titulo="Varianza Explicada"):
  """
  Función para graficar la Varianza Explicada (VE)

  Parámetros:
    ve: Array con los valores de VE para cada componente principal.
    n_componentes: Número de componentes principales.
    titulo: Título del gráfico.

  Retorno:
    None. Muestra un gráfico de la VE.
  """

  plt.figure(figsize=(10, 6))
  plt.plot(range(1, n_componentes + 1), ve, "bo-")
  plt.xlabel("Componente Principal")
  plt.ylabel("Varianza Explicada")
  plt.title(titulo)
  plt.grid()
  plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose

def plot_elementos_serie_temporal(df, start_date, end_date, filt=None):
    # Realizar la descomposición estacional
    result = seasonal_decompose(df[start_date:end_date]['value'], 
                                model='multiplicative', 
                                filt=filt, 
                                extrapolate_trend='freq')

    # Calcular el componente cíclico
    cyclic = result.trend / result.seasonal

    # Crear una figura y ejes
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    # Graficar los cuatro componentes
    df[start_date:end_date]['value'].plot(ax=axes[0], label='Serie original')
    result.trend.plot(ax=axes[1], label='Tendencia', color='red')
    result.seasonal.plot(ax=axes[2], label='Estacionalidad', color='green')
    cyclic.plot(ax=axes[3], label='Cíclico', color='purple')
    result.resid.plot(ax=axes[4], label='Residuo', color='orange')

    # Ajustar la leyenda y etiquetas
    for ax in axes:
        ax.legend()
        ax.set_ylabel('Valor')

    plt.tight_layout()
    plt.show()


def plot_modelos_regression_metrics(y_true, y_pred):
    # Calcular las métricas
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Crear el gráfico de dispersión con la línea de regresión
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='blue', label='Datos reales vs. Predicciones')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Línea de regresión')
    plt.title('Gráfico de Regresión')
    plt.xlabel('Valores Verdaderos')
    plt.ylabel('Valores Predichos')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    # Anotar las métricas en el gráfico
    metrics_text = f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.2f}"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.show()


def graficar_series_temporales_X_periodos_anuales(df, figsize=(15, 10), fontsize=12):

  if not isinstance(df, pd.DataFrame):
    raise TypeError("df debe ser un DataFrame")
  if df.empty:
    raise ValueError("df no puede estar vacío")

  # Obtener la columna de fecha (índice)
  fecha_col = df.index.name

  # Validar que el índice sea datetime
  if not pd.api.types.is_datetime64_dtype(df.index):
    raise TypeError("El índice del DataFrame debe ser de tipo datetime")
  # Obtener los años únicos

  anios = df.index.year.unique()
  # Crear la figura

  fig, axes = plt.subplots(nrows=len(anios) // 3 + 1, ncols=3, figsize=figsize)
  # Recorrer los años y graficar las series temporales

  for i, anio in enumerate(anios):
    df_anio = df[df.index.year == anio]
    for j, columna in enumerate(df_anio.columns):
      ax = axes[i // 3, i % 3]
      df_anio[columna].plot(ax=ax, marker=".", title=f"{anio} - {columna}")
      ax.set_xlabel("")
      ax.set_ylabel("")
      ax.tick_params(labelsize=fontsize)

  # Eliminar los ejes vacíos
  for ax in axes.ravel():
    if not ax.has_data():
      ax.set_visible(False)

  # Ajustar la figura
  fig.tight_layout()
  plt.show()



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

def pinta_distribucion_numericas(df, columnas_numericas, relativa=False, mostrar_valores=False):
    num_columnas = len(columnas_numericas)
    num_filas = (num_columnas // 2) + (num_columnas % 2)

    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten() 

    for i, col in enumerate(columnas_numericas):
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
    
    plt.show();


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


def generar_raincloud_plot_para_num_cat(dataframe):
    # Filtrar columnas numéricas o categóricas ordinales
    columnas_numericas = dataframe.select_dtypes(include=['number']).columns
    columnas_categoricas_ordinales = [col for col in dataframe.columns if dataframe[col].dtype.name == 'category']

    if len(columnas_numericas) == 0 and len(columnas_categoricas_ordinales) == 0:
        print("No se encontraron columnas numéricas o categóricas ordinales en el dataframe.")
        return

    # Contador de subplots
    subplots = 0
    num_subplots = len(columnas_numericas) + len(columnas_categoricas_ordinales)
    num_filas = np.ceil(num_subplots / 3)  # Calcular el número de filas necesario para los subplots

    # Crear raincloud plots para cada columna
    fig, axs = plt.subplots(int(num_filas), 3, figsize=(20, 4*num_filas))

    for col in columnas_numericas:
        #nueva figura cada 3 en numericas
        ax = axs[subplots // 3, subplots % 3]
        pt.RainCloud(x=col, data=dataframe, ax=ax)
        ax.set_title(f'Raincloud Plot para {col}')
        subplots += 1

    for col in columnas_categoricas_ordinales:
        #nueva figura cada 3  categoricas
        ax = axs[subplots // 3, subplots % 3]
        pt.RainCloud(x=col, data=dataframe, ax=ax)
        ax.set_title(f'Raincloud Plot para {col}')
        subplots += 1

    # Eliminar los subplots no utilizados
    for ax in axs.flat[subplots:]:
        ax.remove()

    plt.tight_layout()
    plt.show();



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
        plt.show();

def plot_grouped_histograms_num(df, num_col1, num_col2, group_size):
    num_unique = len(df)
    for i in range(0, num_unique, group_size):
        subset_df = df.iloc[i:i+group_size]
        
        plt.figure(figsize=(10, 6))
        for index, row in subset_df.iterrows():
            sns.histplot(subset_df[num_col1], kde=True, label=str(row[num_col2]))
        
        plt.title(f'Histograms of {num_col1} for {num_col2} (Group {i//group_size + 1})')
        plt.xlabel(num_col1)
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
    Devuelve una lista con las columnas numéricas del df cuya correlación con la columna designada por "target_col" sea superior en valor absoluto al valor dado por "umbral_corr".

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
    max_graficas_por_grupo = 4

    # Dividir en grupos según el número máximo de gráficas
    len(columnas_para_pintar) // max_graficas_por_grupo
    # En un alista de comprension, iteramos en rango desde 0 hasta el numero de columnas a pintar, por cada grupo maximo establecido
    # creando graficas con columnas maxi de i+ grupo max establecido ( ejem: '0 hasta 0+6)
    columnas = [columnas_para_pintar[i:i+max_graficas_por_grupo] for i in range(0, len(columnas_para_pintar), max_graficas_por_grupo)]

    # iteramos por i y por valor 'umbral_corr' establecido a cada grupo en cada iteración,  creeando pair plots para cada grupo,
    for i, grupo in enumerate(columnas):
        sns.pairplot(df, vars=grupo, hue=target_col)
        plt.suptitle(f"Group {i}", y=1.02)# creo nombres de grupo un poco por encima de y, para que no se superponga con la gráfica
        plt.show()
    
    return "Las columnas con un mayor umbral_corr al establecido son", columnas_umbral_mayor



def plot_hist_features_num_bivariante(df: pd.DataFrame, target_col: int = "", 
                                  columns: list = []) -> list:
    
    # Comprueba si 'target_col' es una columna numérica válida en el df
    if target_col and (target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col])):
        print(f"Error: '{target_col}' no es una columna numérica válida en el df.")
        return None
    
    # Comprueba si 'columns' es una lista válida de strings
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        print("Error: 'columns' debería ser una lista de strings.")
        return None
    
    # Si 'columns' está vacío, utiliza todas las columnas numéricas en el df
    if not columns:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Realiza pruebas estadísticas para cada columna
    selected_columns = []
    for col in columns:
        if col == target_col:
            continue
        _, pvalue = f_oneway(df[col], df[target_col])
        if pvalue < 0.05:  # Umbral de significancia
            selected_columns.append(col)
    
    if not selected_columns:
        print("Ninguna columna cumple con las condiciones especificadas para trazar.")
        return None
    
    # Definir colores personalizados
    num_colors = len(selected_columns)
    colors = sns.color_palette("viridis", num_colors)
    target_color = "red"
    
    # Histogramas
    for idx, num_col in enumerate(selected_columns):
        plt.figure(figsize=(6, 6))
        sns.histplot(data=df, x=target_col, hue=num_col, multiple="stack", kde=True, palette=[colors[idx]], legend=False)
        sns.histplot(data=df, x=target_col, color=target_color, kde=True, label=target_col, legend=False)
        plt.title(f"Histograma para {target_col} por {num_col}")
        plt.legend(labels=[num_col], loc='upper right', title='Columna')
        plt.show();
    #kernel interractivo para quitar letras mientras pinta
    plt.ion()
    
    return selected_columns;

def categorical_correlation_heatmap(df):
    # Calcula la matriz de contingencia
    contingency_matrix = pd.DataFrame(np.zeros((len(df.columns), len(df.columns))), columns=df.columns, index=df.columns)
    for col1 in df.columns:
        for col2 in df.columns:
            contingency_matrix.loc[col1, col2] = pd.crosstab(df[col1], df[col2]).values.ravel()[0]
    
    # Calcula el coeficiente de contingencia
    chi2, _, _, _ = chi2_contingency(contingency_matrix)
    contingency_coefficient = np.sqrt(chi2 / (len(df) * min(len(df.columns)-1, len(df.index)-1)))
    
    # Crea el heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(contingency_coefficient, index=df.columns, columns=df.columns), annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Heatmap de Correlación Categórica (Coeficiente de Contingencia)")
    plt.show()

    return contingency_coefficient;


    
def get_features_cat_regression(df: pd.DataFrame, target_col: str, pvalue: float = 0.05) -> list:
    """
    Esta función recibe un df y dos argumentos adicionales: 'target_col' y 'pvalue'.
    
    Parámetros:
    - df: DataFrame de pandas.
    - target_col: Nombre de la columna que actuará como el objetivo para un modelo de regresión.
    - pvalue: Valor de p umbral para la significancia estadística (por defecto es 0.05).
    
    Devuelve:
    - Una lista con las columnas categóricas cuya relación con 'target_col' es estadísticamente significativa.
    - None si hay errores en los parámetros de entrada.
    """
    # Comprueba si 'target_col' es una columna numérica válida en el df
    if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' no es una columna numérica válida en el df.")
        return None
    
    # Comprueba si 'pvalue' es un float válido
    if not isinstance(pvalue, float):
        print("Error: 'pvalue' debería ser un float.")
        return None
    
    # Identifica las columnas categóricas
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Comprueba si hay columnas categóricas
    if not cat_columns:
        print("Error: No se encontraron columnas categóricas en el df.")
        return None
    
    # Realiza pruebas estadísticas y filtra columnas basadas en el valor de p
    selected_columns = []
    for cat_col in cat_columns:
        contingency_table = pd.crosstab(df[cat_col], df[target_col])
        
        # Elige la prueba apropiada según la cardinalidad
        if len(contingency_table) > 2:
            _, p, _, _ = chi2_contingency(contingency_table)
        else:
            _, p = f_oneway(*[df[target_col][df[cat_col] == category] for category in df[cat_col].unique()])
        
        if p < pvalue:
            selected_columns.append(cat_col)
    
    return selected_columns


def plot_features_cat_regression(df: pd.DataFrame, target_col: str = "", 
                                  columns: list = [], pvalue: float = 0.05, 
                                  with_individual_plot: bool = False) -> list:
    """
    Esta función recibe un df y varios argumentos opcionales para visualizar y analizar la relación
    entre variables categóricas y una columna objetivo en un modelo de regresión.

    Parámetros:
    - df: DataFrame de pandas.
    - target_col: Nombre de la columna que actuará como el objetivo para un modelo de regresión.
    - columns: Lista de nombres de columnas categóricas a considerar (por defecto, todas las numéricas).
    - pvalue: Valor de p umbral para la significancia estadística (por defecto es 0.05).
    - with_individual_plot: Booleano que indica si se deben incluir gráficos individuales para cada columna (por defecto es False).

    Devuelve:
    - Una lista con las columnas seleccionadas que cumplen con las condiciones de significancia.
    - None si hay errores en los parámetros de entrada.
    """
    # Comprueba si 'target_col' es una columna numérica válida en el df
    if target_col and (target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col])):
        print(f"Error: '{target_col}' no es una columna numérica válida en el df.")
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
    
    # Si 'columns' está vacío, utiliza todas las columnas numéricas en el df
    if not columns:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filtra columnas basadas en pruebas estadísticas
    selected_columns = get_features_cat_regression(df, target_col, pvalue)
    selected_columns = list(set(selected_columns) & set(columns))
    
    if not selected_columns:
        print("Ninguna columna cumple con las condiciones especificadas para trazar.")
        return None
    
    # Histogramas
    for cat_col in selected_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=target_col, hue=cat_col, multiple="stack", kde=True)
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

def mi_logaritmo(datos):
    base_logaritmo= "e"

    if base_logaritmo == 'e':
        # Usa np.log1p para el logaritmo natural de 1 más el valor (maneja los ceros)
        return np.log1p(datos)
    else:
        # Añade una pequeña constante (por ejemplo, 1) para otras bases para manejar los ceros
        return np.log(datos + 1e-8)  # Ajusta la constante según sea necesario
    
def bivariante_analysis_scatter_num(df, target_column):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_columns.remove(target_column)
    
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=col, y=target_column)
        plt.title(f'Bivariate Analysis: {col} vs {target_column}')
        plt.xlabel(col)
        plt.ylabel(target_column)
        plt.show()



def bivariante_analysis_hist_num(df, target_column):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_columns.remove(target_column)
    
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))
        plt.hist(df[col], alpha=0.5, bins=20)
        plt.title(f'Bivariate Analysis: {col} vs {target_column}')
        plt.xlabel(col)
        plt.ylabel(target_column)
        plt.show()


from scipy.stats import ttest_ind

def test_t_student(df):
    # Definir la variable objetivo
    target = 'target'

    # Filtrar las características numéricas del df
    features_num = df.select_dtypes(include=['number']).columns.tolist()

    # Realizar la prueba t de Student para cada característica numérica
    resultados_pruebas = {}
    for feature in features_num:
        # Realizar la prueba t de Student
        t_statistic, p_value = ttest_ind(df[feature], df[target])
        resultados_pruebas[feature] = {'t_statistic': t_statistic, 'p_value': p_value}

    # Imprimir los resultados
    for feature, result in resultados_pruebas.items():
        print(f"--COLUMNA: {feature}")
        print(f"Estadística de prueba t: {result['t_statistic']}")
        print(f"Valor p: {result['p_value']}")
        print("Significativo: ", "Sí" if result['p_value'] < 0.05 else "No")
    return

    
