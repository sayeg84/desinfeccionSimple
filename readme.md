# desinfeccionSimple

Este repositorio contiene el código correspondiente a una simulación de desinfección de una superficie mediante caminantes aleatorios independientes en Python3

## Dependencias

Todas las dependencias necesarias están listadas en `requirements.txt` y pueden instalarse con pip utilizando el siguiente comando en una terminal:

`pip/pip3 install -r requirements.txt`

Usar `pip` o `pip3` dependiendo de si se utiliza `python` o `python3` en la terminal.

## Descripción

Podemos describir brevemente los archivos principales

|Nombre|Descripción|
|:-:|:-:|
|`principal.py`| Código base de la simulación. Correrlo intentará hacer una simulación para ver que pase las pruebas|
|`graficas.py`| Script para hacer gráficas específicas estadísticas de la simulación. Las guarda en la carpeta `figures` |
|`animaciones.py`| Script para hacer animaciones en video de la simulación haciendo cuadro por cuadro y animando con `ImageIO` . Ver script para tener detalles. Las guarda en la carpeta `figures` |
|`animaciones_matplotlib.py`| Script para hacer animaciones en video de la simulación usando solo `matplotlib`. Generalmente más rápido que `animaciones.py` . Ver script para tener detalles. Las guarda en la carpeta `figures` |

El archivo `principal.py` contiene el código necesario e importante para correr la simulación. Correrlo como script generará un video correspondiente a la simulación.

El archivo `graficas.py` hace simulaciones más específicas y genera gráficas estadísticas de ellas.