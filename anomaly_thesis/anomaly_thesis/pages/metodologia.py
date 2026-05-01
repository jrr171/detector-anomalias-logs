"""
pages/metodologia.py — Documentación académica del sistema
"""
import streamlit as st


def render():
    st.markdown('<div class="section-title">ℹ️ Marco Metodológico</div>', unsafe_allow_html=True)

    st.markdown("""
    Esta sección documenta la base teórica y técnica del sistema, útil para redactar el
    capítulo de **Materiales y Métodos** de tu tesis.
    """)

    tabs = st.tabs(["📖 Marco teórico", "🔧 Pipeline técnico", "📐 Métricas", "📚 Referencias"])

    with tabs[0]:
        st.markdown("""
        ## Detección de anomalías
        La detección de anomalías (*anomaly detection*) es una rama del aprendizaje automático que busca
        identificar patrones que difieren significativamente del comportamiento esperado en un conjunto de datos.
        Estos patrones se denominan **anomalías**, **outliers** o **valores atípicos**.

        ### Tipos de anomalías
        | Tipo | Descripción | Ejemplo |
        |------|-------------|---------|
        | **Puntual** | Un registro individual es inusual | Temperatura de 200°C en sensor normal |
        | **Contextual** | Es anómalo según su contexto | Temperatura de 30°C en invierno |
        | **Colectiva** | Un grupo de registros es inusual | Múltiples transacciones rápidas |

        ### Enfoque no supervisado
        Este sistema usa detección **no supervisada**, lo que significa que no requiere datos
        etiquetados previos. Los algoritmos aprenden la estructura normal de los datos y
        detectan desviaciones automáticamente.
        """)

    with tabs[1]:
        st.markdown("""
        ## Pipeline de procesamiento

        ```
        CSV de entrada
            ↓
        1. PREPROCESAMIENTO
           • Detección automática de separadores y encoding
           • Extracción de columnas numéricas
           • Imputación de valores nulos (mediana)
           • Detección y parsing de columnas de fecha
            ↓
        2. NORMALIZACIÓN
           • RobustScaler (resistente a outliers)
           • Rango de salida: no acotado, centrado en mediana
            ↓
        3. DETECCIÓN (paralela por algoritmo)
           • Autoencoder MLP     → error de reconstrucción
           • Isolation Forest    → profundidad de aislamiento
           • LOF                 → factor de densidad local
           • Z-Score             → desviaciones estándar
           • IQR                 → violaciones del rango IQ
            ↓
        4. ENSEMBLE (votación mayoritaria)
           • Se cuentan los votos de cada algoritmo por registro
           • Anomalía si ≥ mayoría de algoritmos lo marcan
            ↓
        5. ANÁLISIS Y EXPORTACIÓN
           • Estadísticas descriptivas
           • Perfil de anomalías (t-test por columna)
           • Visualizaciones interactivas
           • Exportación CSV
        ```

        ### RobustScaler vs MinMaxScaler
        Se elige **RobustScaler** porque usa la mediana y el IQR en lugar de la media y la desviación estándar,
        lo que lo hace más resistente a los propios outliers que queremos detectar.
        """)

    with tabs[2]:
        st.markdown("""
        ## Métricas del sistema

        ### Umbral de detección
        Se usa el **percentil P** como umbral (configurable). Un percentil del 97 significa que se marcan
        como anomalías el 3% de registros con mayor score de anomalía.

        ### Error de reconstrucción (Autoencoder)
        $$MSE_i = \\frac{1}{n} \\sum_{j=1}^{n} (x_{ij} - \\hat{x}_{ij})^2$$

        Donde $x_{ij}$ es el valor original y $\\hat{x}_{ij}$ es el valor reconstruido por el autoencoder.

        ### Score de anomalía por algoritmo
        | Algoritmo | Métrica de score |
        |-----------|-----------------|
        | Autoencoder | MSE de reconstrucción |
        | Isolation Forest | Negativo del score de aislamiento |
        | LOF | Negativo del factor de outlier local |
        | Z-Score | Máximo z-score por fila |
        | IQR | Número de columnas fuera del rango IQ |

        ### Perfil estadístico de anomalías
        Se aplica un **t-test de dos muestras independientes** por columna para evaluar si las medias
        de normales y anomalías difieren significativamente (p < 0.05).
        """)

    with tabs[3]:
        st.markdown("""
        ## Referencias bibliográficas sugeridas

        1. **Chandola, V., Banerjee, A., & Kumar, V. (2009)**. Anomaly detection: A survey.
           *ACM Computing Surveys*, 41(3), 1–58. https://doi.org/10.1145/1541880.1541882

        2. **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008)**. Isolation forest.
           *2008 Eighth IEEE International Conference on Data Mining*, 413–422.

        3. **Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000)**. LOF: identifying density-based
           local outliers. *ACM SIGMOD Record*, 29(2), 93–104.

        4. **Hinton, G. E., & Salakhutdinov, R. R. (2006)**. Reducing the dimensionality of data with neural
           networks. *Science*, 313(5786), 504–507.

        5. **Pedregosa, F., et al. (2011)**. Scikit-learn: Machine learning in Python.
           *Journal of Machine Learning Research*, 12, 2825–2830.

        6. **Goldstein, M., & Uchida, S. (2016)**. A comparative evaluation of unsupervised anomaly
           detection algorithms for multivariate data. *PLOS ONE*, 11(4).

        ---
        > 💡 **Tip para tu tesis:** Cita estas referencias en tu capítulo de *Estado del Arte* o
        > *Marco Teórico* para fundamentar la elección de cada algoritmo.
        """)
