#pip install ucimlrepov

from obtaindata import readdata as rd
import pandas as pd


data =  pd.DataFrame()


if __name__ == "__main__":
    #code for Apartment for Rent Classified
    idcode=555
   

    data = rd(idcode)

    print(data.info())


