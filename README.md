# Proyecto de Sobremuestreo y Evaluación de Modelos

Este repositorio contiene el código, los datasets y los resultados generados durante el desarrollo del presente trabajo, centrado en técnicas de sobremuestreo y evaluación de modelos de clasificación.  

## Contenido del proyecto

El archivo entregado tiene la siguiente estructura:

├── Datasets/ # Contiene los cinco conjuntos de datos utilizados, en formato CSV
├── Resultados/ # Contiene los resultados obtenidos del proyecto en CSV
├── src/ # Código fuente del proyecto
│ ├── custom_smote.py # Implementación de la variante propuesta de SMOTE
│ ├── experiments.py # Funciones para ejecutar experimentos, optimización de parámetros y evaluación de modelos
│ └── utils.py # Funciones auxiliares utilizadas en el proyecto
└── requirements.txt # Librerías necesarias para ejecutar el proyecto



## Descripción de carpetas y archivos

- **Datasets/**: Contiene los cinco conjuntos de datos utilizados en el proyecto, todos en formato CSV.  
- **Resultados/**: Almacena los resultados obtenidos del proyecto, incluyendo:  
  - Parámetros óptimos encontrados para cada técnica de sobremuestreo.  
  - Parámetros óptimos encontrados para cada modelo.  
  - Métricas de evaluación de cada modelo aplicado a cada dataset.  

- **src/**: Código fuente del proyecto, que incluye:  
  - `custom_smote.py`: Implementación de la variante propuesta de SMOTE.  
  - `experiments.py`: Funciones para ejecutar experimentos, optimización de parámetros y evaluación de modelos.  
  - `utils.py`: Funciones auxiliares utilizadas a lo largo del proyecto.  


- `requirements.txt`: Contiene todas las librerías necesarias para ejecutar el proyecto. Para instalarlas, ejecutar:  

pip install -r requirements.txt
