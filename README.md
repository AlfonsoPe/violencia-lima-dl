# Datos

Los datasets utilizados son públicos y **no se suben** a este repositorio por su tamaño.

## Fuentes

1. **Real-Life Violence Situations (RLVS)**  
   https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset  
   2 000 videos (1 000 Violence / 1 000 NonViolence)

2. **Hockey Fight Videos**  
   Disponible en Kaggle / repositorios académicos públicos.

3. **Movies Fight Detection Dataset**  
   Disponible en Kaggle.

## Cómo obtenerlos

El notebook principal (`notebooks/Proyecto_violencia_Lima.ipynb`) descarga automáticamente los datasets mediante `kagglehub` cuando se ejecuta en Google Colab o en un entorno con acceso a internet.

## Partición oficial

- Train: 70 %
- Validation: 15 %
- Test: 15 %

La partición se realiza **estrictamente a nivel de video** para garantizar cero fuga de datos (Zero Data Leakage).
