import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Valorador Inmobiliario — Rionegro",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fondo */
.stApp {
    background: #F7F4EF;
}

/* Encabezado principal */
.hero {
    background: #1C2B3A;
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem;
    line-height: 1.2;
    color: #F0E6C8;
}
.hero p {
    font-size: 0.9rem;
    color: #9BAFC2;
    margin: 0;
    font-weight: 300;
}

/* Tarjeta de sección */
.section-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid #E8E2D9;
}
.section-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 600;
    color: #1C2B3A;
    margin: 0 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #F0E6C8;
}

/* Resultado */
.result-box {
    background: #1C2B3A;
    border-radius: 12px;
    padding: 2rem 2rem 1.6rem;
    text-align: center;
    margin-top: 1rem;
}
.result-label {
    font-size: 0.72rem;
    color: #9BAFC2;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.6rem;
    font-weight: 500;
}
.result-price {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #F0E6C8;
    line-height: 1;
    margin: 0;
}
.result-divider {
    width: 36px;
    height: 1px;
    background: rgba(240, 230, 200, 0.25);
    margin: 1rem auto;
}
.result-m2 {
    font-size: 1rem;
    color: #C8D5E0;
    margin: 0;
    font-weight: 400;
    letter-spacing: 0.02em;
}
.result-m2-label {
    font-size: 0.82rem;
    color: #9BAFC2;
}
.model-badge {
    display: inline-block;
    background: rgba(46, 95, 74, 0.6);
    color: #A8D5B5;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 999px;
    margin-top: 1.2rem;
    letter-spacing: 0.03em;
}

/* Info pill */
.info-pill {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: #EDF3F0;
    border: 1px solid #C5DDCE;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #2E5F4A;
    margin-bottom: 1rem;
    line-height: 1.5;
}
.info-pill svg { flex-shrink: 0; margin-top: 1px; }

/* Labels de widgets */
label[data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #5A6475 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Inputs y selects */
div[data-baseweb="select"] > div:first-child,
div[data-baseweb="base-input"] {
    border-color: #DDD8D0 !important;
    border-radius: 8px !important;
    background-color: #FDFCFA !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
div[data-baseweb="select"] > div:first-child:focus-within,
div[data-baseweb="base-input"]:focus-within {
    border-color: #1C2B3A !important;
    box-shadow: 0 0 0 3px rgba(28, 43, 58, 0.08) !important;
}

/* Color de texto dentro de inputs */
div[data-baseweb="select"] span,
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div > div,
div[data-baseweb="base-input"] input {
    color: #1C2B3A !important;
}

/* Botón primario */
.stButton > button[data-testid="baseButton-primary"] {
    background: #1C2B3A !important;
    color: #F0E6C8 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    font-size: 0.88rem !important;
    padding: 0.7rem 1.5rem !important;
    transition: background 0.15s ease !important;
    text-transform: uppercase !important;
}
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: #253D54 !important;
    color: #F0E6C8 !important;
}

/* Responsive */
@media (max-width: 640px) {
    .hero { padding: 1.8rem 1.4rem 1.4rem; }
    .hero h1 { font-size: 1.5rem; }
    .result-price { font-size: 2.1rem; }
}

/* Ocultar elementos de Streamlit */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cargar modelo y recursos ──────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    modelo   = joblib.load("mejor_modelo.pkl")
    features = json.load(open("features.json"))
    dist_map = json.load(open("dist_aeropuerto_por_barrio.json"))
    return modelo, features, dist_map

try:
    modelo, feature_cols, dist_por_barrio = cargar_modelo()
    modelo_ok = True
except Exception as e:
    modelo_ok = False
    st.error(f"Error cargando el modelo: {e}")


# ── Encabezado ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Valorador Inmobiliario<br>Rionegro, Antioquia</h1>
    <p>Estimación de precio de venta basada en aprendizaje de máquina · Gradient Boosting</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-pill">
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/>
        <line x1="12" y1="8" x2="12.01" y2="8"/>
    </svg>
    <span>Ingresa las características del inmueble y obtén una estimación del precio de venta en el mercado de Rionegro.
    Los datos provienen de fuentes inmobiliarias locales (Ciencuadras, FincaRaíz, Lonja de Propiedad Raíz).</span>
</div>
""", unsafe_allow_html=True)


# ── Formulario ────────────────────────────────────────────────────────────────
if modelo_ok:

    # Sección 1: Tipo y ubicación
    st.markdown("""
    <div class="section-card">
        <p class="section-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
                <circle cx="12" cy="10" r="3"/>
            </svg>
            Tipo y ubicación
        </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        tipo_inmueble = st.selectbox(
            "Tipo de inmueble",
            options=["apartamento", "casa", "casa_campestre"],
            format_func=lambda x: {"apartamento": "Apartamento", "casa": "Casa", "casa_campestre": "Casa campestre"}[x]
        )
    with col2:
        barrio = st.selectbox(
            "Barrio / Sector",
            options=sorted(dist_por_barrio.keys()),
            format_func=lambda x: x.title()
        )

    col3, col4 = st.columns(2)
    with col3:
        estrato = st.selectbox(
            "Estrato",
            options=[1, 2, 3, 4, 5, 6],
            index=3,
            help="Clasificación socioeconómica en Colombia: 1 (bajo) a 6 (alto). Determina tarifas de servicios públicos y perfil del sector."
        )
    with col4:
        conjunto_cerrado = st.selectbox(
            "Conjunto cerrado",
            options=["si", "no"],
            format_func=lambda x: "Sí" if x == "si" else "No",
            help="Urbanización o edificio con acceso restringido, vigilancia y áreas comunes compartidas."
        )

    dist_aeropuerto = dist_por_barrio.get(barrio, 5.0)
    st.caption(f"Distancia al aeropuerto J.M. Córdova: {dist_aeropuerto} km — asignada automáticamente según el barrio seleccionado.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Sección 2: Características físicas
    st.markdown("""
    <div class="section-card">
        <p class="section-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
                <polyline points="9 22 9 12 15 12 15 22"/>
            </svg>
            Características físicas
        </p>
    """, unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        area_m2 = st.number_input("Área construida (m²)", min_value=20, max_value=5000, value=80, step=5)
    with col6:
        habitaciones = st.number_input("Habitaciones", min_value=1, max_value=15, value=3, step=1)

    col7, col8 = st.columns(2)
    with col7:
        banos = st.number_input("Baños", min_value=1, max_value=10, value=2, step=1)
    with col8:
        parqueaderos = st.number_input("Parqueaderos", min_value=0, max_value=10, value=1, step=1)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Botón de predicción ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predecir = st.button("Estimar precio", use_container_width=True, type="primary")

    if predecir:
        with st.spinner("Calculando estimación..."):
            # Construir vector de entrada
            entrada = {col: 0 for col in feature_cols}

            entrada['area_m2']            = area_m2
            entrada['habitaciones']       = habitaciones
            entrada['banos']              = float(banos)
            entrada['estrato']            = float(estrato)
            entrada['parqueaderos']       = float(parqueaderos)
            entrada['conjunto_cerrado']   = 1 if conjunto_cerrado == "si" else 0
            entrada['dist_aeropuerto_km'] = dist_aeropuerto

            key_tipo = f"tipo_inmueble_{tipo_inmueble}"
            if key_tipo in entrada:
                entrada[key_tipo] = 1

            key_barrio = f"barrio_{barrio}"
            if key_barrio in entrada:
                entrada[key_barrio] = 1

            X_input = pd.DataFrame([entrada])[feature_cols]

            precio_log = modelo.predict(X_input)[0]
            precio_cop = np.expm1(precio_log)
            precio_m2  = precio_cop / area_m2

        precio_fmt    = f"${precio_cop/1e6:,.0f}M COP"
        precio_m2_fmt = f"${precio_m2/1e6:,.2f}M"

        st.markdown(f"""
        <div class="result-box">
            <p class="result-label">Precio estimado de venta</p>
            <p class="result-price">{precio_fmt}</p>
            <div class="result-divider"></div>
            <p class="result-m2">{precio_m2_fmt}<span class="result-m2-label"> / m²</span></p>
            <span class="model-badge">Gradient Boosting &nbsp;·&nbsp; R² = 0.7176 &nbsp;·&nbsp; RMSE = $696M</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        margen = 696_200_000
        low    = max(0, precio_cop - margen)
        high   = precio_cop + margen

        st.info(
            f"**Margen de error estimado (±1 RMSE):** "
            f"${low/1e6:,.0f}M — ${high/1e6:,.0f}M COP"
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#999;'>Proyecto integrador CRISP-DM · Maestría en Ciencia de Datos · UPB · 2025 · "
    "Dataset: mercado inmobiliario Rionegro (web scraping)</small>",
    unsafe_allow_html=True
)
