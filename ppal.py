#pip install ucimlrepov
#
import logging


import pandas as pd


from obtaindata import readdata as rd


#dataframe receive data read it from ucimlrepo
data =  pd.DataFrame()

# Configure basic logging to the console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Get a logger
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    #code for Apartment for Rent Classified
    idcode=555
   

    data = rd(idcode)

    logger.debug("Data obtain from UCIMLrepo:")
    logger.debug(data.info())

    logger.debug("Data describe")
    logger.debug(data.head())

#    print(data.info())


