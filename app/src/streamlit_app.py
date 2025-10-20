import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from unidecode import unidecode
import rasterio as rio
from rasterio.mask import mask
from rasterstats import zonal_stats
from pathlib import Path



# --- Configuración de la página de Streamlit ---
st.set_page_config(layout="wide")
st.title("Las heladas en el Perú: Políticas Públicas - 2025")

# ====== INTRODUCCIÓN DEL PROYECTO ======
st.markdown("""

Este proyecto presenta un **análisis espacial de las temperaturas mínimas en el Perú** con base en datos ráster y capas geográficas oficiales.  
El objetivo es **identificar las zonas más vulnerables a las heladas** y proponer políticas públicas que reduzcan los efectos de las bajas temperaturas sobre la salud y el bienestar de la población, especialmente de **niños menores de 5 años**.

### Fuentes de datos utilizadas

- **Temperatura mínima en el Perú (raster TIFF)**  
  [https://drive.google.com/drive/folders/1kf8Kfuo3EkmcPfQMIyVKPug0FwnzTNHP](https://drive.google.com/drive/folders/1kf8Kfuo3EkmcPfQMIyVKPug0FwnzTNHP)

- **Distritos (Capa vectorial)**  
  [https://github.com/jotikeng1/Raster_Analysis_Public_Policy_Streamlit/raw/refs/heads/main/data/shape_file/DISTRITOS.shp]("https://github.com/jotikeng1/Raster_Analysis_Public_Policy_Streamlit/raw/refs/heads/main/data/shape_file/DISTRITOS.shp)

- **Plan Multisectorial ante Heladas y Friaje 2022 – 2024**  
  [https://sigrid.cenepred.gob.pe/sigridv3/storage/biblioteca/15522_plan-multisectorial-ante-heladas-y-friajes-2022-2024.pdf](https://sigrid.cenepred.gob.pe/sigridv3/storage/biblioteca/15522_plan-multisectorial-ante-heladas-y-friajes-2022-2024.pdf)

### Importancia del análisis

Las **heladas** son uno de los fenómenos climáticos más severos que afectan al altiplano peruano, generando pérdidas agrícolas, afectaciones a la ganadería y problemas de salud en comunidades vulnerables.  
El estudio de la **temperatura mínima del aire** permite identificar los territorios con mayor exposición, fortalecer la gestión preventiva y orientar **acciones multisectoriales** en salud, vivienda y desarrollo social.  

En este proyecto se integran **herramientas de análisis geoespacial (Raster + GeoJSON)** con información de políticas públicas vigentes, buscando aportar evidencia para la **prevención adaptativa frente al cambio climático**.
""")


# Ruta al archivo
# archivo = os.path.join("..", "..", "data", "tmin_raster.tif")

# Ruta absoluta robusta basada en este archivo
archivo = Path(__file__).resolve().parents[2] / "data" / "tmin_raster.tif"

# Ruta de distritos geojson
shapefile_path = os.path.join("https://github.com/jotikeng1/Raster_Analysis_Public_Policy_Streamlit/raw/refs/heads/main/data/shape_file/DISTRITOS.shp")
gdf = gpd.read_file(shapefile_path)



# --- Funciones de Carga de Datos (con caché para velocidad) ---
@st.cache_data
def load_data():
    with rio.open(archivo) as src:
        meta = {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": src.crs.to_string() if src.crs else None,
            "transform": tuple(src.transform),
            "dtype": src.dtypes,
            "scales": getattr(src, "scales", None),
            "offsets": getattr(src, "offsets", None),
        }
        b1 = src.read(1, masked=True).astype("float32")
    return meta, b1



# Abrir el archivo
with rio.open(archivo) as src:
    print("Ancho:", src.width)
    print("Alto:", src.height)
    print("Número de bandas:", src.count)
    print("Sistema de coordenadas:", src.crs)
    print("Transformación (afín):", src.transform)

    # Leer los datos de la primera banda
    banda1 = src.read(1)
    banda2 = src.read(2)
    banda3 = src.read(3)
    banda4 = src.read(4)
    banda5 = src.read(5)

@st.cache_data
def load_map_data():
    gdf = gpd.read_file(shapefile_path)
    return gdf



# promedio de temperaturas minimas en cada distrito 
promedios = []
for b in range(1, 6):
    stats = zonal_stats(gdf, archivo, stats=["mean"], band=b)
    promedios.append([s["mean"] for s in stats])

for i, valores in enumerate(promedios): # Crear columnas para cada banda en el GeoDataFrame
    gdf[f"tmin_banda{i+1}"] = valores

gdf["tmin_promedio_total"] = gdf[[f"tmin_banda{i+1}" for i in range(5)]].mean(axis=1) # Calcular el promedio total de las 5 bandas

print(gdf[["DISTRITO", "tmin_promedio_total"]].head()) # Mostrar resultados con los nombres de distritos


# Merge  oon la data de distritos
with rio.open(archivo) as src:
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)
    
    out_image, out_transform = mask(src, gdf.geometry, crop=True)# Recortar el raster usando el shapefile
    out_meta = src.meta.copy()

out_meta.update({# Actualizar metadata para el nuevo raster recortado
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform
})

with rio.open("recorte.tif", "w", **out_meta) as dest:# Guardar el nuevo raster recortado (opcional)
    dest.write(out_image)


# Abrir y recortar todas las bandas del ráster
with rio.open(archivo) as src:
    out_image, out_transform = mask(src, gdf.geometry, crop=True)
    out_meta = src.meta.copy()

out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform
})

with rio.open("tmin_peru_recortado.tif", "w", **out_meta) as dest:# Guardar el nuevo ráster recortado
    dest.write(out_image)


# Metricas
# Crear una columna de departamento para análisis si no existe
if 'DEPARTAMENTO' not in gdf.columns:
    if 'DEPARTAME' in gdf.columns:
        gdf['DEPARTAMENTO'] = gdf['DEPARTAME']
    elif 'IDDPTO' in gdf.columns:
        # Mapeo completo de códigos de departamento a nombres en Perú
        dpto_map = {
            '01': 'AMAZONAS',
            '02': 'ANCASH',
            '03': 'APURIMAC',
            '04': 'AREQUIPA',
            '05': 'AYACUCHO',
            '06': 'CAJAMARCA',
            '07': 'CALLAO',
            '08': 'CUSCO',
            '09': 'HUANCAVELICA',
            '10': 'HUANUCO',
            '11': 'ICA',
            '12': 'JUNIN',
            '13': 'LA LIBERTAD',
            '14': 'LAMBAYEQUE',
            '15': 'LIMA',
            '16': 'LORETO',
            '17': 'MADRE DE DIOS',
            '18': 'MOQUEGUA',
            '19': 'PASCO',
            '20': 'PIURA',
            '21': 'PUNO',
            '22': 'SAN MARTIN',
            '23': 'TACNA',
            '24': 'TUMBES',
            '25': 'UCAYALI'
        }
        # Aplicar el mapeo, asegurando que siempre hay un nombre de departamento
        gdf['DEPARTAMENTO'] = gdf['IDDPTO'].astype(str).map(lambda x: dpto_map.get(x, f"DPTO_{x}"))

# Análisis Estadístico Básico
print("Estadísticas descriptivas de temperatura promedio:")
stats = gdf['tmin_promedio_total'].describe()
stats.head()




# --- Construimos la tabla de políticas ---
def build_politicas_df():
        return pd.DataFrame({
            "Región": ["Altiplano (heladas)"],
            "Objetivo específico": ["Reducir IRAs por heladas en zonas altoandinas"],
            "Población": ["Distritos con Tmin ≤ p10 situados en regiones: Puno, Cusco, Ayacucho, Huancavelica, Pasco."],
            "Intervención": ["Kits de abrigo internado, kits de abrigo para población vulnerable (niñas y niños de 0–5 años), vacunas contra la influenza."],
            "Costo estimado (S/)": ["La intervención busca beneficiar a 219 691 niños menores de cinco años en las regiones priorizadas de Puno, Cusco, Ayacucho, Huancavelica y Pasco. De acuerdo con el Plan Multisectorial ante Heladas y Friajes 2022–2024, para las tres intervenciones propuestas se destinará un presupuesto total de S/ 5 056 527, equivalente a aproximadamente S/ 23 por niño beneficiario."],
            "KPI": ["Evitar que el 80% de la población menor a 5 años se enferme por casos de IRA (infección respiratoria aguda)."]
        })



# --- Configuración de la página de Streamlit ---
st.set_page_config(layout="wide")
st.title('Las heladas en el Perú: Polìticas Públicas - 2025')



tab1, tab2, tab3, tab4 = st.tabs(["Preparación de Datos y Estadísticas", "Análisis y Visualizaciones", "Propuesta de Política", "Integrantes grupo 7"])


with tab1:
    st.markdown("### Preparación de la data")
    st.write("---")

    # ---------- Vista rápida de distritos ----------
    st.markdown("#### Vista rápida de distritos (primeras filas)")
    st.dataframe(gdf.head(10), use_container_width=True, height=240)

    # ---------- Alinear CRS del vector con el raster ----------
    with rio.open(archivo) as src:
        raster_crs = src.crs
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
        st.info("CRS de distritos reproyectado para coincidir con el CRS del ráster.")
    st.caption(f"CRS distritos: `{gdf.crs}`  |  CRS ráster: `{raster_crs}`")

    st.write("---")
    st.markdown("#### Metadatos del ráster")

    try:
        meta  # comprobar si existe
    except NameError:
        with rio.open(archivo) as src:
            meta = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": src.crs.to_string() if src.crs else None,
                "transform": tuple(src.transform),
                "dtype": src.dtypes,
                "scales": getattr(src, "scales", None),
                "offsets": getattr(src, "offsets", None),
            }

    c1, c2, c3 = st.columns(3)
    c1.metric("Ancho (px)", meta["width"])
    c2.metric("Alto (px)", meta["height"])
    c3.metric("N° de bandas", meta["count"])
    st.code(f"CRS: {meta['crs']}", language="text")
    st.code(f"Transformación (afín): {meta['transform']}", language="text")
    st.code(f"dtype por banda: {meta['dtype']}", language="text")
    st.code(f"scales: {meta.get('scales')} | offsets: {meta.get('offsets')}", language="text")

    st.write("---")
    st.markdown("#### Estadísticas básicas de temperatura mínima promedio")

    # Nombre de columna a usar (ajusta si tu DataFrame tiene otro)
    col_tmn = "tmin_promedio_total"
    if col_tmn not in gdf.columns:
        st.error(f"No se encontró la columna `{col_tmn}` en gdf. Columnas disponibles: {list(gdf.columns)}")
    else:
        serie = gdf[col_tmn]

        # Tabla de estadísticas principales
        stats_main = serie.agg(["count", "mean", "median", "std", "min", "max"]).to_frame(name="valor")
        stats_main.loc["missing"] = serie.isna().sum()
        stats_main.loc["unique distritos"] = len(gdf)

        # Cuantiles útiles
        q = serie.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).rename("valor").to_frame()
        q.index = [f"q{int(i*100)}" for i in q.index]

        st.markdown("**Resumen principal**")
        st.table(stats_main.T)

        st.markdown("**Cuantiles**")
        st.table(q.T)

        # Descargar estadísticas
        stats_out = pd.concat([stats_main, q])
        csv_stats = stats_out.to_csv(encoding="utf-8-sig")
        st.download_button(
            label="Descargar estadísticas (CSV)",
            data=csv_stats.encode("utf-8-sig"),
            file_name="estadisticas_tmin.csv",
            mime="text/csv",
        )



with tab2:
    st.subheader("Exploración de temperaturas mínimas por distrito")

    # --- Parámetros rápidos ---
    colp1, colp2 = st.columns(2)
    with colp1:
        bins = st.slider("N° de bins para el histograma", 10,20,40)
    with colp2:
        top_k = st.slider("Top 15 distritos", 15)

    # ======================
    # 1) Histograma
    # ======================
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(gdf["tmin_promedio_total"], kde=True, bins=bins, ax=ax)
    ax.set_title("Temperatura mínima promedio (°C)")
    ax.set_ylabel("Frecuencia")
    ax.set_xlabel("°C")
    ax.axvline(gdf["tmin_promedio_total"].mean(), color="red", linestyle="--",
               label=f"Media: {gdf['tmin_promedio_total'].mean():.2f}")
    ax.axvline(gdf["tmin_promedio_total"].median(), color="green", linestyle="--",
               label=f"Mediana: {gdf['tmin_promedio_total'].median():.2f}")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # ======================
    # 2) Tops
    # ======================
    c1, c2 = st.columns(2)

    # Top menores
    with c1:
        st.markdown("#### Top 15 distritos con **menor** Tmin promedio")
        top_low = gdf.sort_values(by="tmin_promedio_total").head(top_k)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax = sns.barplot(data=top_low, x="DISTRITO", y="tmin_promedio_total", color="steelblue", ax=ax)
        ax.set_title(f"Top {top_k} — Tmin promedio más baja")
        ax.set_xlabel("Distrito")
        ax.set_ylabel("°C")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        for p in ax.patches:
            ax.text(p.get_x()+p.get_width()/2, p.get_height()+0.1, f"{p.get_height():.1f}°C",
                    ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # Top mayores
    with c2:
        st.markdown("#### Top 15 distritos con **mayor** Tmin promedio")
        top_high = gdf.sort_values(by="tmin_promedio_total", ascending=False).head(top_k)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax = sns.barplot(data=top_high, x="DISTRITO", y="tmin_promedio_total", color="steelblue", ax=ax)
        ax.set_title(f"Top {top_k} — Tmin promedio más alta")
        ax.set_xlabel("Distrito")
        ax.set_ylabel("°C")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        for p in ax.patches:
            ax.text(p.get_x()+p.get_width()/2, p.get_height()+0.1, f"{p.get_height():.1f}°C",
                    ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    st.divider()

    # ======================
    # 3) Mapa coroplético
    # ======================
    st.markdown("#### Mapa coroplético — Tmin promedio por distrito")
    fig, ax = plt.subplots(figsize=(10, 11))
    gdf.plot(column="tmin_promedio_total",
             cmap="RdYlBu_r",   # paleta adecuada para temperatura (frío ↔ caliente)
             legend=True,
             legend_kwds={"label": "Tmin promedio (°C)"},
             linewidth=0.2, edgecolor="white",
             ax=ax)
    ax.set_title("Temperatura mínima promedio por distrito")
    ax.set_axis_off()
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.divider()

    # ======================
    # 4) Tabla resumen + descarga
    # ======================
    st.markdown("#### Tabla resumen distrital")
    tabla_distrital = (gdf[["IDDIST", "DEPARTAMENTO", "PROVINCIA", "DISTRITO", "tmin_promedio_total"]]
                       .copy()
                       .rename(columns={"tmin_promedio_total": "Tmin_promedio(°C)"}))
    tabla_distrital["Tmin_promedio(°C)"] = tabla_distrital["Tmin_promedio(°C)"].round(2)

    st.dataframe(tabla_distrital, use_container_width=True, height=320)

    csv_bytes = tabla_distrital.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="Descargar tabla distrital (CSV)",
        data=csv_bytes,
        file_name="tabla_tmin_distrital.csv",
        mime="text/csv",
    )
                       
       

with tab3:
    st.markdown("### Propuesta de Política Pública para prevenir IRA en niños menores a 5 años")

    politicas = build_politicas_df()

    # Opción 1: tabla estática que envuelve texto (se ve mejor para celdas largas)
    st.table(politicas)

    # Opción 2: tabla interactiva con scroll horizontal (si prefieres)
    # st.dataframe(politicas, use_container_width=True, height=220)

    # Botón de descarga
    csv_bytes = politicas.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="⬇️ Descargar CSV de políticas",
        data=csv_bytes,
        file_name="politicas_publicas_tmin.csv",
        mime="text/csv",
    )


with tab4:
        st.markdown("### Integrantes grupo 7:")
        st.markdown(
        """
        - Arellano Morán, Grabiel  
        - Magno Fabián, Eduardo 
        - Contreras Valenzuela, Romel  
        - Moreno Gómez, Lizeth 
        """
        )