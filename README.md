# 📈 Predicción Multidimensional de Carga de Trabajo - Random Forest Contable/Tributario

Este proyecto implementa un modelo predictivo basado en Machine Learning (**Random Forest Regressor**) para anticipar y proyectar la carga de trabajo diaria (cantidad de tareas y modificaciones en sistemas del negocio) en una firma de asesoría contable y tributaria. 

A diferencia de los enfoques estándar de series temporales, este modelo utiliza un enfoque, cruzando el historial transaccional de los desarrolladores o usuarios con un calendario fiscal exógeno para capturar con precisión los picos de trabajo causados por vencimientos de impuestos.

---

## 🎯 El Problema de Negocio
En el sector contable y tributario, la carga de trabajo operativa no es lineal ni puramente estacional respecto al calendario tradicional; se rige por decisiones humanas y **vencimientos fiscales externos** (IVA, Retención en la Fuente, Renta, etc.). 

Un análisis basado únicamente en promedios semanales sufriría de "puntos ciegos" masivos, siendo incapaz de predecir picos repentinos de actividad (por ejemplo, saltos drásticos de 20 a más de 80 tareas en un solo día). Este modelo soluciona ese problema mediante **Ingeniería de Características (Feature Engineering) avanzada** enfocada en el contexto del negocio.

---

## 🚀 Características Principales

* **Arquitectura Multifuente Separada:** Mantiene la integridad de los datos separando el archivo de transacciones (`historial_cambios.csv`) de la fuente maestra de impuestos (`calendario_tributario.csv`), garantizando un mantenimiento ágil si ocurren prórrogas estatales de última hora.
* **Tratamiento de Datos Dinámico (Campos Multivalor):** Limpia y formatea de manera automática cadenas de texto dinámicas con comas o fechas repetidas (ej. `"18/6/2026, 18/06/2026"`), evitando caídas del script en producción.
* **Ingeniería de Regresores Temporales:**
    * `dias_para_vencimiento`: Distancia matemática exacta en días hacia el deadline más cercano.
    * `presion_vencimiento`: Bandera binaria (0 o 1) que alerta al modelo cuando se entra en la ventana crítica de entrega (0 a 5 días).
    * `Lag Features & Rolling Windows`: Memoria del comportamiento del día anterior, del mismo día de la semana pasada y una media móvil suavizada de estabilidad de 7 días.
* **Predicción Recursiva:** Un bucle autorregresivo que proyecta los próximos 7 días simulando el futuro a partir de sus propias predicciones y recalculando el contexto fiscal dinámicamente.
* **Detección Automática de Picos:** Implementación de procesamiento de señales con `scipy.signal.find_peaks` para rotular visualmente las anomalías y techos operativos históricos y proyectados.

---

## 📊 Arquitectura de Datos (Formatos Soportados)

El script consume de forma nativa dos archivos en la ruta local o entorno `/content/`:

1.  **`historial_cambios.csv`** (Historial operativo):
    * `fecha_modificacion`: Estampa de tiempo (soporta formatos AM/PM tradicionales y mixtos).
    * `tipo_cambio`: Categoría de la tarea (convertida automáticamente a *One-Hot Encoding / Dummies*).
    * `nombre_historial`: Identificador del usuario (para calcular usuarios activos por día).

2.  **`calendario_tributario.csv`** (Calendario fiscal maestro):
    * `Tipo de impuesto`: Nombre del tributo regulado.
    * `Fecha`: Fecha límite del vencimiento fiscal.

---

## 🛠️ Requisitos e Instalación

Para ejecutar este proyecto de forma local o en entornos interactivos como Jupyter Notebook / Google Colab, asegúrate de contar con Python 3.8+ y las siguientes librerías:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
