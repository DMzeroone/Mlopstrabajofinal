import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def readdata(idcode):
    
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

        return data
    except ModuleNotFoundError:
        logger.debug("La librería 'ucimlrepo' no está instalada.")
        # Puedes añadir aquí un comando para instalarla, por ejemplo:
        import subprocess
        subprocess.check_call(["pip", "install", "ucimlrepo"])


