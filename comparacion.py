"""
pages/inicio.py — Página de bienvenida
"""
import streamlit as st


def render():
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0;">
        <div style="font-size:4rem;">🔬</div>
        <h1 style="font-size:2.5rem; font-weight:700; background: linear-gradient(135deg,#60a5fa,#a78bfa);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0.5rem 0;">
            AnomalyIQ
        </h1>
        <p style="color:#94a3b8; font-size:1.1rem; max-width:600px; margin:0 auto;">
            Sistema inteligente de detección de anomalías en datos tabulares.<br>
            Compara múltiples algoritmos de Machine Learning y genera reportes académicos.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem;">🧠</div>
            <div class="metric-label">Algoritmos</div>
            <div class="metric-value">5</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem;">⚡</div>
            <div class="metric-label">Ensemble</div>
            <div class="metric-value">✓</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem;">📊</div>
            <div class="metric-label">Métricas</div>
            <div class="metric-value">10+</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem;">📥</div>
            <div class="metric-label">Exportable</div>
            <div class="metric-value">CSV</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">🚀 ¿Cómo usar el sistema?</div>', unsafe_allow_html=True)
        pasos = [
            ("1️⃣", "Configurar", "Elige algoritmos y parámetros en el panel lateral"),
            ("2️⃣", "Subir CSV", "Ve a **Análisis** y sube tu archivo de datos"),
            ("3️⃣", "Analizar", "Presiona el botón y espera el procesamiento"),
            ("4️⃣", "Interpretar", "Revisa métricas, gráficas y tabla de anomalías"),
            ("5️⃣", "Exportar", "Descarga los resultados en CSV para tu tesis"),
        ]
        for icono, titulo, desc in pasos:
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:0.8rem; margin-bottom:0.8rem;
                        background:#1a2035; border-radius:8px; padding:0.8rem;">
                <span style="font-size:1.4rem;">{icono}</span>
                <div>
                    <div style="font-weight:600; color:#e2e8f0;">{titulo}</div>
                    <div style="color:#94a3b8; font-size:0.85rem;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-title">🤖 Algoritmos incluidos</div>', unsafe_allow_html=True)
        algos = [
            ("🧠", "Autoencoder (MLP)", "Red neuronal encoder-decoder. Aprende la estructura normal de los datos.", "Alta dimensionalidad"),
            ("🌲", "Isolation Forest", "Árboles aleatorios que aíslan outliers eficientemente.", "Datos mixtos"),
            ("📍", "LOF", "Compara densidad local de cada punto con sus vecinos.", "Anomalías contextuales"),
            ("📊", "Z-Score", "Detecta valores estadísticamente extremos por columna.", "Distribuciones normales"),
            ("📐", "IQR", "Rango intercuartílico, robusto ante distribuciones sesgadas.", "Outliers univariados"),
        ]
        for icono, nombre, desc, uso in algos:
            st.markdown(f"""
            <div class="algo-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:600; color:#e2e8f0;">{icono} {nombre}</span>
                    <span style="background:#1e3a5f; color:#93c5fd; font-size:0.7rem;
                                 padding:2px 8px; border-radius:999px;">{uso}</span>
                </div>
                <div style="color:#94a3b8; font-size:0.82rem; margin-top:0.3rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#475569; font-size:0.8rem; padding:1rem;">
        AnomalyIQ · Proyecto de Tesis · Detección de Anomalías con Machine Learning
    </div>
    """, unsafe_allow_html=True)
