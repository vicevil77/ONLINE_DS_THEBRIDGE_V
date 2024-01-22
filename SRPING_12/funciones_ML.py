#funcion para que te diga los mejores hiperparametros para ajustar la generalziacion de un modelo de regresion lienal
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error

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

# Mostrar métricas en el conjunto de prueba
print("MAE test:", mae_test)
print("MSE test:", mse_test)
print("RMSE test:", rmse_test)
print("R2 test:", r2_test)