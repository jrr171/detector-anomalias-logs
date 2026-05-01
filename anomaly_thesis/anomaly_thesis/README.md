# 🔬 AnomalyIQ — Sistema de Detección de Anomalías

Proyecto de tesis de Machine Learning no supervisado para la detección de anomalías en datos tabulares.

## 🚀 Instalación y ejecución

```bash
# 1. Clonar o descomprimir el proyecto
cd anomaly_thesis

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

## 📁 Estructura del proyecto

```
anomaly_thesis/
├── app.py                    # Aplicación principal Streamlit
├── requirements.txt          # Dependencias
├── README.md                 # Este archivo
│
├── core/                     # Lógica de negocio
│   ├── detector.py           # 5 algoritmos de detección + ensemble
│   ├── metricas.py           # Estadísticas, correlaciones, perfiles
│   └── almacenamiento.py     # Histórico CSV y log de análisis
│
└── pages/                    # Páginas de la interfaz
    ├── inicio.py             # Página de bienvenida
    ├── analisis.py           # Módulo principal de análisis
    ├── comparacion.py        # Comparación entre cargas
    ├── historial.py          # Gestión del historial
    └── metodologia.py        # Documentación académica
```

## 🤖 Algoritmos implementados

| Algoritmo | Tipo | Ventaja |
|-----------|------|---------|
| Autoencoder (MLP) | Deep Learning | Captura relaciones no lineales complejas |
| Isolation Forest | Ensemble | Eficiente, alta dimensionalidad |
| LOF | Densidad | Anomalías contextuales |
| Z-Score | Estadístico | Simple, interpretable |
| IQR | Estadístico | Robusto a distribuciones asimétricas |
| **Ensemble (votación)** | **Meta-modelo** | **Combina todos los anteriores** |

## ⚙️ Configuración

Desde el panel lateral de la app puedes configurar:
- **Percentil de anomalías**: qué tan estricto es el detector (90–99%)
- **Iteraciones del modelo**: precisión del autoencoder
- **Algoritmos activos**: cuáles comparar en cada análisis

## 📥 Exportación

- Dataset completo con columnas de etiquetas por algoritmo
- Solo las anomalías detectadas por el ensemble
- Log de todos los análisis históricos

## 📚 Referencia académica principal

Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey.
*ACM Computing Surveys*, 41(3), 1–58.
