# Análisis Exploratorio de Datos (EDA) - Aprendizaje Automatico en la Nube - Universidad de Medellin

## 👥 Equipo de trabajo

| Integrante | Usuario de Git |
|-------------|------|
| **Camila Andrea Millán Chaparro** | CamilaMillan-C |
| **Daniel Morales** | DMzeroone |
| **Mauricio Taborda** | pirripipa |

## Descripción general

En esta entrega se realiza un **Análisis Exploratorio de Datos (EDA)** sobre el conjunto de datos **“Apartment for Rent Classified”**, obtenido del repositorio de **UCI Machine Learning**.  
El objetivo principal del análisis es explorar la base de datos y comprender las relaciones entre variables como precio, tamaño, número de habitaciones, ubicación y condiciones del anuncio, con el fin de identificar patrones o tendencias que sirvan como base para modelos predictivos o segmentación.

---

## EDA

1. **Carga de datos:**  
   Se importó el dataset desde el repositorio de UCI utilizando la librería `ucimlrepo`.

2. **Revisión inicial:**  
   Se realizó una exploración básica para comprender la estructura del conjunto de datos, el número de registros y tipos de variables.

3. **Limpieza y transformación de datos:**  
   El conjunto de datos original se copió a `datostransformados`, sobre el cual se realizaron las siguientes acciones:
   - Conversión de las columnas `bathrooms`, `bedrooms` y `square_feet` a valores numéricos.  
   - Codificación binaria de variables como `has_photo` (creando `has_photo_bin`) y `fee` (creando `has_fee_bin`).  
   - Aplicación de **One-Hot Encoding** sobre variables categóricas (`category`, `pets_allowed`, `price_type`), etc 
   - Sustitución de valores nulos en `pets_allowed` por “No”.  
   - Eliminación de columnas generadas erróneamente (`cat_Gym`, `pets_Monthly`, `pri_Los Angeles`, `pri_VA`), esto debido a que
  los comentarios de detalle de las casas incluian comas al leer el csv, generaba distorsiones en la informacion.

   Estas transformaciones garantizaron la preparación de los datos para el análisis estadístico.

4. **Visualización y análisis descriptivo:**  
   Se elaboraron gráficos para observar distribuciones, tendencias y posibles outliers en las variables numéricas.

5. **Correlaciones:**  
   Se construyó una matriz de correlación entre variables numéricas, representada mediante un mapa de calor.  
   Este análisis permite identificar relaciones directas o inversas entre variables como el precio, el tamaño o el número de habitaciones.


---



