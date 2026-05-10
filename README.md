# Valorador Inmobiliario — Rionegro, Antioquia

Aplicación web de predicción de precios de inmuebles residenciales en el municipio de Rionegro (Antioquia, Colombia), desarrollada como proyecto integrador de la Maestría en Ciencia de Datos de la Universidad Pontificia Bolivariana.

## Descripción

El modelo predice el precio de venta de apartamentos, casas y casas campestres en Rionegro a partir de sus características físicas y geoespaciales, siguiendo la metodología CRISP-DM en sus seis fases completas.

## Metodología

**CRISP-DM** — 6 fases:
1. Entendimiento del negocio
2. Entendimiento de los datos
3. Preparación de los datos
4. Modelamiento
5. Evaluación
6. Despliegue

## Dataset

- **Fuente:** Web scraping de portales inmobiliarios (Ciencuadras, FincaRaíz, Lonja de Propiedad Raíz, Metrocuadrado)
- **Registros:** 1.897 inmuebles
- **Municipio:** Rionegro, Antioquia, Colombia
- **Uso autorizado** por el docente de la materia

## Modelo

| Parámetro | Valor |
|---|---|
| Algoritmo | Gradient Boosting (scikit-learn) |
| MAE | $245.1M COP |
| RMSE | $696.2M COP |
| R² | 0.7176 |
| Variable objetivo | log1p(precio_cop) |

### Variables predictoras
- `area_m2` — Área construida (m²)
- `habitaciones` — Número de habitaciones
- `banos` — Número de baños
- `estrato` — Estrato socioeconómico (1–6)
- `parqueaderos` — Número de parqueaderos
- `conjunto_cerrado` — Pertenece a conjunto cerrado (0/1)
- `dist_aeropuerto_km` — Distancia al Aeropuerto J.M. Córdova (km)
- `tipo_inmueble_*` — One-Hot Encoding del tipo de inmueble
- `barrio_*` — One-Hot Encoding del barrio

## Estructura del repositorio

```
├── app.py                          # Aplicación Streamlit
├── mejor_modelo.pkl                # Modelo serializado
├── features.json                   # Lista de features en orden
├── dist_aeropuerto_por_barrio.json # Distancias por barrio
├── requirements.txt                # Dependencias
├── dataset_rionegro.csv            # Dataset original
└── CRISP_DM_Rionegro_Completo.ipynb # Notebook completo (Fases 3–5)
```

## Instalación local

```bash
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud

1. Haz fork o push de este repositorio a GitHub
2. Ingresa a [streamlit.io](https://streamlit.io) y conecta tu cuenta de GitHub
3. Selecciona el repositorio y el archivo `app.py`
4. Haz clic en **Deploy**

## Autor

**Daniel Jesús Castillo Botero**  
Maestría en Ciencia de Datos — Universidad Pontificia Bolivariana  
Materia: Aprendizaje de Máquina
