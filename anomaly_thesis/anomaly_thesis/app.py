import streamlit as st

st.set_page_config(
    page_title="AnomalyIQ - Sistema de Detección de Anomalías",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

.metric-card {
    background: linear-gradient(135deg, #1a2035 0%, #1e2d4a 100%);
    border: 1px solid #2d4a7a;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #60a5fa;
    border-left: 4px solid #60a5fa;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

.anomaly-badge {
    display: inline-block;
    background: #7f1d1d;
    color: #fca5a5;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}

.normal-badge {
    display: inline-block;
    background: #14532d;
    color: #86efac;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}

.algo-card {
    background: #1a2035;
    border: 1px solid #2d4a7a;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}

div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

.sidebar-logo {
    text-align: center;
    padding: 1rem;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    border-bottom: 1px solid #2d4a7a;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Inicializar session state
if "analizado" not in st.session_state:
    st.session_state.analizado = False
if "ultima_carga" not in st.session_state:
    st.session_state.ultima_carga = None
if "df_resultado" not in st.session_state:
    st.session_state.df_resultado = None
if "metricas" not in st.session_state:
    st.session_state.metricas = None
if "scores_por_algo" not in st.session_state:
    st.session_state.scores_por_algo = {}

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🔬 AnomalyIQ</div>', unsafe_allow_html=True)
    st.markdown("**Sistema de Detección de Anomalías**")
    st.markdown("*Proyecto de Tesis - Machine Learning*")
    st.divider()

    pagina = st.radio(
        "Navegación",
        ["🏠 Inicio", "📊 Análisis", "📈 Comparación de Modelos", "📋 Historial", "ℹ️ Metodología"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Configuración del Modelo**")

    umbral_percentil = st.slider("Percentil de anomalías (%)", 90, 99, 97, 1,
                                  help="Define qué tan estricto es el detector. 97 = top 3% más raro")

    max_iter = st.selectbox("Iteraciones del modelo", [50, 100, 200, 500], index=1,
                             help="Más iteraciones = más precisión pero más lento")

    algoritmos = st.multiselect(
        "Algoritmos a comparar",
        ["Autoencoder (MLP)", "Isolation Forest", "LOF", "Z-Score", "IQR"],
        default=["Autoencoder (MLP)", "Isolation Forest", "LOF"]
    )

# ==========================================
# PÁGINAS
# ==========================================
if pagina == "🏠 Inicio":
    from pages import inicio
    inicio.render()

elif pagina == "📊 Análisis":
    from pages import analisis
    analisis.render(umbral_percentil, max_iter, algoritmos)

elif pagina == "📈 Comparación de Modelos":
    from pages import comparacion
    comparacion.render()

elif pagina == "📋 Historial":
    from pages import historial
    historial.render()

elif pagina == "ℹ️ Metodología":
    from pages import metodologia
    metodologia.render()
