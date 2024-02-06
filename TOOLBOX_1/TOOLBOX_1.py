import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import seaborn as sns
from sklearn.feature_selection import f_regression

#PRIMERA
def funcion_describe(df):

    resultado = pd.DataFrame()
    """
    Obtiene información sobre porcentaje de cardinalidad, el tipo de datos, los valores únicos y el porcentaje de valores NaN en cada columna de un DataFrame.

    Args:
        df: El DataFrame del que se quiere obtener la información.

    Returns:
        Un diccionario con la información de cada columna.
    """
    for col in df.columns:
        datos = {}
        datos['%_cardinalidad'] = round(df[col].nunique() / len(df) * 100, 2)
        datos['Tipo_dato'] = df[col].dtype
        datos['valores_unicos'] = df[col].nunique()
        datos['%_NaN'] = round(df[col].isna().mean() * 100, 2)
        resultado[col] = pd.Series(datos)
    return resultado.transpose()



#SEGUNDA    
def funcion_categorias(df):
    
    """
    Obtiene información sobre el tipo de categoria de cada columna de un DataFrame.

    Args:
        df: El DataFrame del que se quiere obtener la información.

    Returns:
        Un diccionario con la información de cada columna.
    """

    resultado = pd.DataFrame()
    for col in df.columns:
        datos = {}
        if pd.api.types.is_numeric_dtype(df[col]):
            datos['Categoria'] = 'numerica continua' if df[col].nunique() > 10 else 'numerica discreta'
        else:
            datos['Categoria'] = 'categorica ordinal' if df[col].nunique() > 2 else 'categorica nominal'
   
        resultado[col] = pd.Series(datos)
    return resultado.transpose()





#TERCERA    
def get_features_num_regression(df, target_col,umbral_corr, pvalue=None):
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
    columnas_correlacionadas = df.columns[abs(df.corr()[target_col]) >= umbral_corr].tolist()

    # Filtramos por significancia si se proporciona un p-value, agrupando columnas con su p_values
    if pvalue is not None:
        columnas_correlacionadas = [col for col, p_val in zip(columnas_correlacionadas, p_values) if p_val <= pvalue]

    return f"Las columnas numéricas con |valor superior| al valor {umbral_corr} aportado en la variable 'umbral_corr' son:{columnas_correlacionadas}"




#CUARTA 
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
    # En un alista de comprension, iteramos en rango desde 0 hasta el numero de columnas a pintar, por cada grupo maximo establecido
    # creando graficas con columnas maxi de i+ grupo max establecido ( ejem: '0 hasta 0+6)
    columnas = [columnas_para_pintar[i:i+max_graficas_por_grupo] for i in range(0, len(columnas_para_pintar), max_graficas_por_grupo)]

    # iteramos por i y por valor 'umbral_corr' establecido a cada grupo en cada iteración,  creeando pair plots para cada grupo,
    for i, grupo in enumerate(columnas):
        sns.pairplot(df, vars=grupo, hue=target_col)
        plt.suptitle(f"Group {i}", y=1.02)# creo nombres de grupo un poco por encima de y, para que no se superponga con la gráfica
        plt.show()
    
    return "Las columnas con un mayor umbral_corr al establecido son", columnas_umbral_mayor




#QUINTA 
def get_features_cat_regression(df, target, cardin= 0, pvalue=0.05):
    # Comprobar si 'target' es una columna numérica continua
    if target not in df.columns or not pd.api.types.is_numeric_dtype(df[target]):
        print(f"La columna '{target}' no es una variable numérica continua.")
        return None

    # comprobar y obtener columnas categóricas con cardinalidad superior a la establecida
    columnas_cat = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])
                           and df[col].nunique()/len(df)*100 > cardin]

    if not columnas_cat:
        print("No existen columnas categóricas en el DataFrame")
        return None
    
     # Declarar variables fuera del bucle
    columnas_significativas = []
    corr = None  # Inicializar a None para el caso en que no haya ninguna columna significativa
    pv = None
    chi2_stat = None
    chi2_pvalue = None
    ttest_pvalue = None
    contingency_table = None
  
    # Realizar pruebas de correlación y almacenar columnas estadisticamente  significativas
    for col in columnas_cat:
        # Comprobar si 'col' es una columna categórica
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Advertencia: La columna '{col}' es de tipo categórico, se ignorará en el análisis.")
            continue
       
        # Realizar prueba de correlación de Pearson
        corr, pv = pearsonr(df[col], df[target])

        # Realizar prueba chi-cuadrado
        contingency_table = pd.crosstab(df[col], df[target])
        chi2_stat, chi2_pvalue, _, _ = chi2_contingency(contingency_table)

        # Realizar prueba t de Student
        _, ttest_pvalue = f_regression(df[col], df[target])

        # Seleccionar el método de prueba en base al p-value
        if pv < pvalue or chi2_pvalue < pvalue or ttest_pvalue < pvalue:
            columnas_significativas.append((col, 'pearson' if pv < pvalue else 'chi2_test' if chi2_pvalue < pvalue else 't_test'))
     
    return columnas_significativas



#SEXTA
#el parámetro 'wirh_indivual_plot, controla si debe generar o no histogramas individuales para cada combinación de variables categóricas y la columna objetivo con relacion significativa.
# si esta en False solo devuelve una lista de columnas categoricas o este caso numericas con significacion  y si es True las pintara
def plot_features_cat_regression(df, target, columns=None, pvalue=0.05, with_individual_plot=False):
    # Si la lista de columnas no se proporciona, seleccionar las variables numéricas del DataFrame
    if not columns:
        columns = df.select_dtypes(include='number').columns.tolist()

    # Inicializar la lista de columnas significativas
    columnas_cat = []

    # Iterar sobre las columnas
    for col in columns:
        # Verificar que los datos en la columna son numéricos
        if pd.to_numeric(df[col]).isna().any():
            print(f"Advertencia: La columna '{col}' contiene valores no numéricos, se ignorará en el análisis.")
            continue

        # Realizar prueba de chi-cuadrado
        contingency_table = pd.crosstab(df[col], df[target])
        chi2_stat, chi2_pvalue, _, _ = chi2_contingency(contingency_table)

        # Seleccionar el método de prueba en base al p-value
        if chi2_pvalue < pvalue:
            columnas_cat.append(col)

            # Generar histogramas individuales si se requiere
            if with_individual_plot:
                plt.figsize=(10,5)
                sns.histplot(data=df, x=col, hue=target, multiple="stack", bins=20,)
                plt.title(f'Histograma para {col} agrupado por {target}')
                plt.show()

    return "Las columnas seleccionadas son:", columnas_cat



