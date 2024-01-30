#funcion para que te diga los mejores hiperparametros para ajustar la generalziacion de un modelo de regresion lienal
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

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

# Ejemplo de uso
# Suponiendo que test_X es una Series o DataFrame de pandas
test_X_reshaped = reshape_test_data(test_X)

#funcion para sacar MAE sin sklearn
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

#funcion paea sacar el MAPE sin skleran
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100