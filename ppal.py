#pip install ucimlrepov
#codigo de https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified
#
import logging
import os


import pandas as pd
import numpy as np
from prefect import flow, task
import statsmodels.api as sm
from scipy import stats


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')


#from obtaindata import readdata as rd


#dataframe receive data read it from ucimlrepo
data =  pd.DataFrame()

#path:dir path where csv is ubicated.
#file:filename csv dataset
#idcode:codigo en el repositorio del dataset utilizado.

path = "./data/"
file = "dataset.csv" 
idcode = 555

# Configure basic logging to the console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Get a logger
logger = logging.getLogger(__name__)


def existdata(file):
    #path = path + file  

    if(os.path.isfile(path+file)):
        return True
    
    return False


#def readdata(idcode,file):
#@task
def extract_data(idcode,file):
    #si el archio existe, lo leo, ya debe estar depurado, sino, lo descargo y lo limpio.
    
    if (existdata(file)):
     
        try:
            #leerlo
            main_df = pd.read_csv(path+file, sep=";")


        except FileNotFoundError:
            logger.debug("CSV no encontrado")
        except pd.errors.EmptyDataError:
            logger.debug("Error: el archivo '{path}' esta vacio o sin datos.")
        except pd.errors.ParserError as e:
            logger.debug("Error parsing el archivo CSV  '{path}': {e}")
        except Exception as e:
            logger.debug("Un error inesperado ocurrio durante la lectura '{path}': {e}")

    else:

        try:
            from ucimlrepo import fetch_ucirepo
            # fetch dataset 
            apartment_for_rent_classified = fetch_ucirepo(id=555) 
            
            # data (as pandas dataframes) 
            main_df = apartment_for_rent_classified.data.features
        
    #        print("X",X)
    #        print("/n")
        
            # variable information 
            logger.debug("features from aparment_for_rent_Classified")
            logger.debug(apartment_for_rent_classified.variables) 


            #cleandata(data)

            
        except ModuleNotFoundError:
            logger.debug("La librería 'ucimlrepo' no está instalada.")
            # Puedes añadir aquí un comando para instalarla, por ejemplo:
            import subprocess
            subprocess.check_call(["pip", "install", "ucimlrepo"])
    return main_df

#@task
def cleaning(main_df):
    
    cleaned_df = main_df.copy()
    if not(existdata(file)):
        #si filtramos por price_type se logra mas consistencia, solo hay 4 registros monthly y monthly|weekly
        cleaned_df = main_df[main_df['price_type'] == 'Monthly']

        cleaned_df['bathrooms'] = pd.to_numeric(main_df['bathrooms'], errors='coerce')
        cleaned_df['bedrooms'] = pd.to_numeric(main_df['bedrooms'], errors='coerce')
        cleaned_df['square_feet'] = pd.to_numeric(main_df['square_feet'], errors='coerce')


        #has_foto a binario, es irreelante si tiene foto o es thumbnail:
        mapeo_foto = {'Yes': 1, 'Thumbnail': 1, 'No': 0}
        cleaned_df['has_photo_bin'] = main_df['has_photo'].map(mapeo_foto)
        del cleaned_df['has_photo']

        mapeo_fee = {'Yes': 1,'No': 0}
        cleaned_df['has_fee_bin'] = main_df['fee'].map(mapeo_fee)
        del cleaned_df['fee']

        #onehot encoding

        cleaned_df = pd.get_dummies(cleaned_df, columns=['category'], prefix='cat', drop_first=True, dtype=int)
        #datostransformados = pd.get_dummies(datostransformados, columns=['category'], prefix='cat', drop_first=True)


        cleaned_df['pets_allowed'].fillna('No',inplace=True)
        cleaned_df = pd.get_dummies(cleaned_df, columns=['pets_allowed'], prefix='pets', drop_first=True, dtype=int)


        #cambiando de unixstamp a time normal.
        cleaned_df['fecha'] = pd.to_datetime(cleaned_df['time'], unit='s')

        # Formatear a dd-mm-yyyy
        cleaned_df['fecha_formateada'] = cleaned_df['fecha'].dt.strftime('%d-%m-%Y')
        del cleaned_df['fecha']

        #error al leer los datos, borrar columnas que no deberian estar creadas, posiblemente por comas en las columnas title o body\
        try:
            '''
            del datostransformados['cat_Gym']
            del datostransformados['pets_Monthly']
            del datostransformados['pri_Los Angeles']
            del datostransformados['pri_VA']
            '''

            cleaned_df.to_csv(path+"dataset.csv",sep=";")
            #datostransformados.to_csv(path+"dataset.csv")
        except FileNotFoundError:
            logger.debug("CSV no encontrado")
        except pd.errors.EmptyDataError:
            logger.debug("Error: el archivo '{path}' esta vacio o sin datos.")
        except pd.errors.ParserError as e:
            logger.debug(f"Error parsing el archivo CSV  '{path}': {e}")
        except Exception as e:
            logger.debug(f"Un error inesperado ocurrio durante la lectura '{path}': {e}")
    
    return cleaned_df
        


#@task
def preprocessing(cleaned_df):
    #dataset con datos borrados
    #logger.debug(cleaned_df.head())s


    
    final_df=cleaned_df[['bathrooms', 'bedrooms', 'price','square_feet','latitude','longitude','fecha_formateada','has_photo_bin',
                              'has_fee_bin','cat_housing/rent/apartment','cat_housing/rent/commercial/retail','cat_housing/rent/condo',
                              'cat_housing/rent/home', 'cat_housing/rent/other','cat_housing/rent/short_term','pets_Cats,Dogs',
                              'pets_Cats,Dogs,None','pets_Dogs','pets_No'
    ]].copy()
 
    
    
    
    final_df.dropna(inplace=True)
    

    # Análisis inicial de ciudades
    '''
    print("=== ANÁLISIS DE CIUDADES ===")
    print(f"Número total de ciudades únicas: {final_df['cityname'].nunique()}")
    print("Top 10 ciudades con más registros:")
    print(final_df['cityname'].value_counts().tail(10))
    
    '''
    return final_df
    #return cleaned_df


#@task
def training(final_df):

    #mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #http://186.121.46.71:5000/#/experiments
    mlflow.set_tracking_uri("http://186.121.46.71:5000")

    logger.debug("tracking URI: '{mlflow.get_tracking_uri()}'")

    mlflow.search_experiments()

    experiment_name = "Apartment_Price_Prediction_Square_Price"
    mlflow.set_experiment(experiment_name)

   # 3. Preparar datos para el modelo
    X = final_df[['square_feet']]  # Variable independiente   
    y = final_df['price']          # Variable dependient

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name=experiment_name):

        #4. Entrenar modelo de regresión lineal simple
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        # 5. Coeficientes del modelo
        print("\n=== MODELO DE REGRESIÓN LINEAL ===")
        print(f"Intercepto (β₀): ${modelo.intercept_:.2f}")
        print(f"Coeficiente (β₁): ${modelo.coef_[0]:.4f} por sq ft")
        

        # 6. Predicciones
        y_pred = modelo.predict(X_test)

        # 7. Métricas de Evaluación
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)


        # Registrar parámetros
        mlflow.log_param("model_type", "Linear Regression Square x Price")
        mlflow.log_param("features_used", list(X.columns))
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])
        
        # Registrar métricas
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # Registrar el modelo
        mlflow.sklearn.log_model(modelo, "model")
        
        # Registrar el scaler
        #mlflow.sklearn.log_model(scaler, "scaler")

        # 9. Pruebas de Significancia Estadística
        print("\n=== SIGNIFICANCIA ESTADÍSTICA ===")

        # Calcular errores estándar y valores t
        n = len(X_train)
        p = 1  # número de predictores

        # Error estándar de los residuales
        residuales = y_test - y_pred
        sse = np.sum(residuales ** 2)
        mse_modelo = sse / (n - p - 1)

        # Matriz de covarianza
        X_with_intercept = np.column_stack([np.ones(len(X_train)), X_train])
        cov_matrix = mse_modelo * np.linalg.inv(X_with_intercept.T @ X_with_intercept)

        # Errores estándar de los coeficientes
        se_intercept = np.sqrt(cov_matrix[0, 0])
        se_slope = np.sqrt(cov_matrix[1, 1])

        # Valores t
        t_intercept = modelo.intercept_ / se_intercept
        t_slope = modelo.coef_[0] / se_slope

        # Valores p
        p_intercept = 2 * (1 - stats.t.cdf(np.abs(t_intercept), n - p - 1))
        p_slope = 2 * (1 - stats.t.cdf(np.abs(t_slope), n - p - 1))

        print(f"Intercepto - Valor t: {t_intercept:.4f}, Valor p: {p_intercept:.4f}")
        print(f"Pendiente - Valor t: {t_slope:.4f}, Valor p: {p_slope:.4f}")

        if p_slope < 0.05:
            print("✅ La relación es estadísticamente significativa (p < 0.05)")
        else:
            print("❌ La relación NO es estadísticamente significativa")

        # 10. Intervalos de Confianza
        confianza = 0.95
        grados_libertad = n - p - 1
        t_critico = stats.t.ppf((1 + confianza) / 2, grados_libertad)

        ic_slope_inf = modelo.coef_[0] - t_critico * se_slope
        ic_slope_sup = modelo.coef_[0] + t_critico * se_slope

        print(f"\nIntervalo de confianza {confianza*100}% para la pendiente:")
        print(f"(${ic_slope_inf:.4f} por sq ft, ${ic_slope_sup:.4f} por sq ft)")


    model_path=final_df.copy()
    return model_path




#@flow
def data_rent():
    #connection = db_connection()
    main_df = extract_data(idcode,file)
    cleaned_df = cleaning(main_df)
    final_df = preprocessing(cleaned_df)
    model_path = training(final_df)
    #deploy(model_path)

if __name__ == "__main__":
    #code for Apartment for Rent Classified
    data_rent()
    '''
    data_rent.serve(
        name="data_rent",
        cron="*/2 * * * *",
        description="Machine learning pipeline to train our model.",
        tags=["data-rent"]
    )
    '''
    '''
    data = rd(idcode,file)

    logger.debug("Data obtain from UCIMLrepo:")
    logger.debug(data.info())

    logger.debug("Data describe")
    logger.debug(data.describe())

    #fecha formateada.
    logger.debug("Data fecha")
    logger.debug(data['fecha_formateada'])
    '''


    '''
    # 11. Predicciones para Nuevos Valores
    def predecir_precio(modelo, metros_cuadrados):
     """Función para predecir precio dado los metros cuadrados"""
        precio_predicho = modelo.predict([[metros_cuadrados]])[0]
        return precio_predicho

       # Ejemplos de predicción
        tamanos_ejemplo = [300, 500, 750, 1000, 1200]
        print("\n=== PREDICCIONES DE EJEMPLO ===")
        for tamano in tamanos_ejemplo:
            precio = predecir_precio(modelo, tamano)
            print(f"• {tamano} sq ft → ${precio:,.2f}")
    '''
    
#    print(data.info())

 
