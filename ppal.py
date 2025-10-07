#pip install ucimlrepov
#codigo de https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified
#
import logging
import os


import pandas as pd

from prefect import flow, task


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
@task
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

@task
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



        #datostransformados = pd.get_dummies(datostransformados, columns=['price_type'], prefix='pri', drop_first=True, dtype=int)

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
        


@task
def preprocessing(cleaned_df):
    #dataset con datos borrados
    #logger.debug(cleaned_df.head())s


    
    final_df=cleaned_df[['bathrooms', 'bedrooms', 'price','square_feet','latitude','longitude','fecha_formateada','has_photo_bin',
                              'has_fee_bin','cat_housing/rent/apartment',
                              'cat_housing/rent/commercial/retail','cat_housing/rent/condo','cat_housing/rent/home',
                              'cat_housing/rent/other','cat_housing/rent/short_term','pets_Cats,Dogs',
                              'pets_Cats,Dogs,None','pets_Dogs','pets_No'
    ]].copy()
 
    
    
    
    final_df.dropna(inplace=True)
    


    #return final_df
    return cleaned_df


@task
def training(final_df):
    model_path=final_df.copy()
    return model_path




@flow
def data_rent():
    #connection = db_connection()
    main_df = extract_data(idcode,file)
    cleaned_df = cleaning(main_df)
    final_df = preprocessing(cleaned_df)
    model_path = training(final_df)
    #deploy(model_path)

if __name__ == "__main__":
    #code for Apartment for Rent Classified


    data_rent.serve(
        name="data_rent",
        cron="*/2 * * * *",
        description="Machine learning pipeline to train our model.",
        tags=["data-rent"]
    )
    
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

    
#    print(data.info())

 
