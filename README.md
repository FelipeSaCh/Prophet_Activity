# 📈 Predicción Multidimensional de Carga de Trabajo - Random Forest Contable/Tributario

Este proyecto implementa un motor predictivo avanzado basado en Machine Learning (**Random Forest Regressor**) diseñado específicamente para anticipar, modelar y mitigar la variabilidad operativa diaria en firmas de asesoría contable y tributaria. 

A diferencia de los enfoques convencionales de series temporales (como ARIMA o Prophet), este modelo aprovecha un enfoque **híbrido y exógeno**, correlacionando el historial transaccional de actividades de los colaboradores con el calendario fiscal obligatorio para capturar con precisión los picos de trabajo inducidos por vencimientos normativos (IVA, Retención en la Fuente, Renta, etc.).

---

## 🎯 El Problema de Negocio (Puntos Ciegos Tradicionales)
En la práctica contable, la carga de trabajo no sigue una estacionalidad lineal ni se rige puramente por días laborables comunes. Está condicionada por **plazos de cumplimiento estatales** rígidos que cambian dinámicamente debido a prórrogas o decretos de última hora.

Un análisis predictivo clásico basado en promedios móviles o suavizado exponencial sufriría de "puntos ciegos" críticos, incapaz de prever transiciones abruptas (por ejemplo, incrementos súbitos de 15 a más de 80 tareas diarias). Este modelo resuelve esta limitación mediante **Ingeniería de Características (Feature Engineering)** de contexto normativo.

---

## 🚀 Arquitectura y Capacidades No Visibles en Versiones Básicas

El código de producción (`BetaTest.ipynb`) implementa una serie de mecanismos analíticos de control que trascienden la simple regresión predictiva:

### 1. Sistema de Control Temporal Multizona (Doble Alerta Estadística)
El modelo no solo proyecta un valor numérico, sino que evalúa la carga en base a límites estadísticos históricos dinámicos:
*   🔴 **Zona Crítica de Sobrecarga (Rojo - Percentil 85):** Identifica días donde la carga supera el P85 histórico. Alertas automatizadas etiquetan visualmente estos picos.
*   🟡 **Fase de Mitigación / Alerta Temprana (Amarillo):** El sistema proyecta la ventana crítica con **1 a 2 días de anticipación** antes de un pico de sobrecarga, permitiendo a la gerencia redistribuir recursos preventivamente.
*   🔵 **Zona de Inactividad Operativa Anómala (Azul - Percentil 15 Hábiles):** Una métrica clave que evalúa caídas atípicas de actividad únicamente de **Lunes a Viernes**. Esto permite diagnosticar de forma automatizada parálisis operativas, fallos en la ingesta de datos o cuellos de botella en sistemas de información del negocio.

### 2. Motor de Predicción Recursiva Autorregresiva
Para predecir el horizonte de los próximos 7 días reales, el modelo utiliza un bucle de simulación iterativo:
1.  Predice el día $t+1$ utilizando variables de retraso de la historia real.
2.  Actualiza dinámicamente las variables dinámicas de retraso (*Lag Features* de 1 día, 7 días y la Media Móvil Suavizada de 7 días) alimentando sus propias proyecciones anteriores al set de predictores para $t+2$.
3.  Calcula en cada iteración la distancia matemática exacta al vencimiento fiscal más cercano en el calendario exógeno de forma automatizada.

### 3. Evaluación de Rendimiento bajo División Temporal Estricta
Para simular el comportamiento real en producción, el script segmenta los datos cronológicamente (dejando los últimos 7 días como set de prueba) en lugar de usar K-Fold aleatorio, lo cual fugaría información temporal (*Data Leakage*):
*   **MAE (Error Absoluto Medio):** Cuantifica la desviación media en número de tareas del negocio.
*   **RMSE (Raíz del Error Cuadrático Medio):** Penaliza los errores de gran magnitud para asegurar estabilidad ante picos.
*   **wMAPE (Error Porcentual Absoluto Medio Ponderado):** Métrica principal de negocio, que evita la división por cero cuando hay días de nula actividad.
*   **Precisión General Ajustada ($100 - wMAPE 	imes 100$):** Indicador directo del nivel de confianza del modelo para la toma de decisiones directivas.

### 4. Desglose Operativo por Responsables (Visibilidad de Procesos)
El script incluye un motor de tabulación cruzada (`pd.crosstab`) enfocado en la transparencia de rendimiento del último periodo activo:
*   Mapea dinámicamente columnas transaccionales como `RESPONSABLE` y `tipo_cambio`.
*   Genera un gráfico de barras horizontales apiladas que se adapta en altura según el número de colaboradores activos.
*   Implementa un algoritmo de etiquetado dinámico interno centrado por segmento, evitando la visualización redundante de valores nulos o en cero, y coronando la barra con la suma total de tareas.

---

## 🛠️ Estructura de Directorios Automatizada (Google Drive)
El script interactúa nativamente con Google Drive (`/content/drive/MyDrive/Prophet_Activity`) y garantiza la persistencia estructurada de reportes en alta resolución (300 DPI) utilizando marcas de tiempo localizadas para la zona horaria de Colombia (`America/Bogota`). 

Al ejecutarse, crea automáticamente el siguiente árbol de carpetas dentro de la ruta del proyecto:
```
Prophet_Activity/
│
├── datasets/
│   └── fechas_impuestos (1).csv             # Calendario tributario exógeno
│
└── Control temporal de multizona/          # Vista 1: Reporte integral de alertas
└── Zonas Críticas/                         # Vista 2: Prevención de cuellos de botella
└── Monitor de Inactividad/                 # Vista 3: Alertas de parálisis operativa
└── Porcentajes de distribucion/            # Vista 4: Distribución porcentual (Pie Chart)
└── Desglose por responsables/              # Vista 5: Rendimiento de colaboradores
```

---

## 📊 Especificación de la Ingeniería de Características (Variables de Entrenamiento)

El Random Forest es entrenado con una matriz de variables multidimensionales que capturan comportamiento temporal y normativo:

| Variable | Tipo | Descripción |
| :--- | :--- | :--- |
| `dia_semana` | Categórico (0-6) | Día de la semana (Lunes a Domingo). |
| `dia_mes` | Numérico (1-31) | Día del mes en curso. |
| `es_sabado` / `es_domingo` | Binaria (0 o 1) | Control explícito de inactividad de fin de semana. |
| `es_inicio_mes` | Binaria (0 o 1) | Alerta de presión por cierres de mes tradicionales (Días 1 al 5). |
| `dias_para_vencimiento` | Numérico entero | Distancia en días reales hasta el plazo tributario más cercano. |
| `presion_vencimiento` | Binaria (0 o 1) | Flag crítico activo si la distancia al impuesto es de 0 a 5 días. |
| `tareas_ayer` | Numérico (*Lag 1*) | Número de tareas ejecutadas en el día operativo anterior. |
| `tareas_hace_7d` | Numérico (*Lag 7*) | Carga operativa registrada en el mismo día de la semana anterior. |
| `media_movil_7d` | Numérico (*Rolling*) | Media móvil suavizada de las últimas 7 jornadas de trabajo. |
| `usuarios_activos` | Numérico entero | Número de colaboradores únicos con actividad registrada el día de evaluación. |
| `tipo_` (Dummies) | Binario (One-Hot) | Desglose por tipos de tareas operativas del sistema contable. |

---

## 🖥️ Requisitos de Software

Instalación rápida para entornos de producción y analíticos:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy pytz
```
