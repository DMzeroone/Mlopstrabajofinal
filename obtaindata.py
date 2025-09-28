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
        print(apartment_for_rent_classified.variables) 

        return data
    except ModuleNotFoundError:
        print("La librería 'ucimlrepo' no está instalada.")
        # Puedes añadir aquí un comando para instalarla, por ejemplo:
        import subprocess
        subprocess.check_call(["pip", "install", "ucimlrepo"])


