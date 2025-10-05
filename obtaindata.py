import os
import logging

import pandas as pd

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


path = "./data/"

def existdata(file):
    #path = path + file  

    if(os.path.isfile(path+file)):
        return True
    
    return False
    
def cleandata(datos):

    datostransformados = datos.copy()

    datostransformados['bathrooms'] = pd.to_numeric(datos['bathrooms'], errors='coerce')
    datostransformados['bedrooms'] = pd.to_numeric(datos['bedrooms'], errors='coerce')
    datostransformados['square_feet'] = pd.to_numeric(datos['square_feet'], errors='coerce')


    #has_foto a binario, es irreelante si tiene foto o es thumbnail:
    mapeo_foto = {'Yes': 1, 'Thumbnail': 1, 'No': 0}
    datostransformados['has_photo_bin'] = datos['has_photo'].map(mapeo_foto)
    del datostransformados['has_photo']

    mapeo_fee = {'Yes': 1,'No': 0}
    datostransformados['has_fee_bin'] = datos['fee'].map(mapeo_fee)
    del datostransformados['fee']

    #onehot encoding

    datostransformados = pd.get_dummies(datostransformados, columns=['category'], prefix='cat', drop_first=True, dtype=int)
    #datostransformados = pd.get_dummies(datostransformados, columns=['category'], prefix='cat', drop_first=True)


    datostransformados['pets_allowed'].fillna('No',inplace=True)
    datostransformados = pd.get_dummies(datostransformados, columns=['pets_allowed'], prefix='pets', drop_first=True, dtype=int)


    datostransformados = pd.get_dummies(datostransformados, columns=['price_type'], prefix='pri', drop_first=True, dtype=int)

    #error al leer los datos, borrar columnas que no deberian estar creadas, posiblemente por comas en las columnas title o body\

    del datostransformados['cat_Gym']
    del datostransformados['pets_Monthly']
    del datostransformados['pri_Los Angeles']
    del datostransformados['pri_VA']

    try:
        datostransformados.to_csv(path+"dataset.csv")

    except FileNotFoundError:
        logger.error("CSV no encontrado")
    except pd.errors.EmptyDataError:
        print(f"Error: el archivo '{path}' esta vacio o sin datos.")
    except pd.errors.ParserError as e:
        print(f"Error parsing el archivo CSV  '{path}': {e}")
    except Exception as e:
        print(f"Un error inesperado ocurrio durante la lectura '{path}': {e}")

    return datostransformados


    






def readdata(idcode,file):
    #si el archio existe, lo leo, ya debe estar depurado, sino, lo descargo y lo limpio.
    
    if (existdata(file)):
     
        try:
            #leerlo
            data = pd.read_csv(path+file)


        except FileNotFoundError:
            logger.error("CSV no encontrado")
        except pd.errors.EmptyDataError:
            print(f"Error: el archivo '{path}' esta vacio o sin datos.")
        except pd.errors.ParserError as e:
            print(f"Error parsing el archivo CSV  '{path}': {e}")
        except Exception as e:
            print(f"Un error inesperado ocurrio durante la lectura '{path}': {e}")

    else:

        try:
            from ucimlrepo import fetch_ucirepo
            # fetch dataset 
            apartment_for_rent_classified = fetch_ucirepo(id=555) 
            
            # data (as pandas dataframes) 
            data = apartment_for_rent_classified.data.features
        
    #        print("X",X)
    #        print("/n")
        
            # variable information 
            logger.debug("features from aparment_for_rent_Classified")
            logger.debug(apartment_for_rent_classified.variables) 


            cleandata(data)

            return data
        except ModuleNotFoundError:
            logger.error("La librería 'ucimlrepo' no está instalada.")
            # Puedes añadir aquí un comando para instalarla, por ejemplo:
            import subprocess
            subprocess.check_call(["pip", "install", "ucimlrepo"])
    
    return data


