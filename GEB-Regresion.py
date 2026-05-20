import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Análisis GEB", layout="wide")
st.title("📈 Análisis de Regresores: Grupo Energía de Bogotá (GEB)")
st.markdown("Este dashboard analiza la relación entre el precio de la acción de GEB y dos variables macroeconómicas clave.")

# Definir fechas (Últimos 6 meses para asegurar > 90 registros)
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=180)

# ===== FUNCIÓN PARA GENERAR DATOS DE EJEMPLO =====
def generate_sample_data(n_days=120):
    """Genera datos simulados realistas usando datos históricos reales de USD/COP y Brent"""
    
    # Intenta descargar datos reales de USD/COP y Brent
    try:
        print("Descargando datos históricos reales...")
        
        # Descargar USD/COP real
        usd_cop_data = yf.download("COP=X", start=start_date, end=end_date, progress=False)['Close']
        
        # Descargar Brent real
        brent_data = yf.download("BZ=F", start=start_date, end=end_date, progress=False)['Close']
        
        # Tomar los últimos n_days registros que tengan datos completos
        df_temp = pd.concat([usd_cop_data, brent_data], axis=1).dropna()
        df_temp.columns = ['USD_COP', 'Brent']
        
        if len(df_temp) >= n_days:
            df_temp = df_temp.tail(n_days)
        else:
            n_days = len(df_temp)
        
        # Generar GEB correlacionado con USD/COP y Brent (con base histórica)
        # Normalizar los datos para mejor correlación
        usd_norm = (df_temp['USD_COP'] - df_temp['USD_COP'].mean()) / df_temp['USD_COP'].std()
        brent_norm = (df_temp['Brent'] - df_temp['Brent'].mean()) / df_temp['Brent'].std()
        
        # GEB correlacionado: base + correlación con USD + correlación con Brent
        geb = 3000 + 200 * usd_norm + 100 * brent_norm + np.random.normal(0, 100, len(df_temp))
        geb = np.abs(geb)  # Asegurar valores positivos
        
        df_temp['GEB'] = geb
        df_temp = df_temp[['GEB', 'USD_COP', 'Brent']]
        df_temp.index.name = 'Fecha'
        
        return df_temp
        
    except Exception as e:
        print(f"No se pudieron descargar datos reales. Generando datos simulados: {e}")
        
        # Si falla, generar datos completamente simulados pero realistas
        dates = pd.date_range(end=end_date, periods=n_days, freq='D')
        
        # Simular series de tiempo realistas
        np.random.seed(42)  # Para reproducibilidad
        
        # Crear variaciones suaves (random walk)
        usd_cop = 3800 + np.cumsum(np.random.normal(0, 20, n_days))
        brent = 80 + np.cumsum(np.random.normal(0, 1.5, n_days))
        
        # GEB correlacionado con USD y Brent
        geb = 3000 + 0.5 * usd_cop + 20 * brent + np.cumsum(np.random.normal(0, 50, n_days))
        geb = np.abs(geb)
        
        df = pd.DataFrame({
            'GEB': geb,
            'USD_COP': np.abs(usd_cop),
            'Brent': np.abs(brent)
        }, index=dates)
        df.index.name = 'Fecha'
        
        return df

# ===== BARRA LATERAL PARA ENTRADA DE DATOS =====
st.sidebar.header("📥 Cargar Datos")

data_source = st.sidebar.radio("Selecciona la fuente de datos:", 
["Subir CSV"])

df = None


# OPCIÓN: Subir archivo CSV
if data_source == "Subir CSV":
    st.sidebar.info("El CSV debe tener columnas: Fecha, GEB, USD_COP, Brent")
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            # Intentar convertir la primera columna a fecha si es posible
            if len(df.columns) > 0:
                # Intentar convertir a datetime
                try:
                    first_col = df.columns[0]
                    df[first_col] = pd.to_datetime(df[first_col])
                except:
                    pass
                df.set_index(df.columns[0], inplace=True)
                df.index.name = 'Fecha'
            st.sidebar.success("✓ Archivo cargado correctamente")
        except Exception as e:
            st.sidebar.error(f"Error al cargar CSV: {e}")


# ===== PROCESAMIENTO Y VISUALIZACIÓN =====
if df is not None and not df.empty:
    # Asegurar que el índice sea datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            pass
    
    # Mostrar información de los datos
    st.subheader("📊 Información de los Datos")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de registros", len(df))
    with col2:
        try:
            fecha_min = df.index.min().strftime('%Y-%m-%d')
            fecha_max = df.index.max().strftime('%Y-%m-%d')
            st.metric("Período", f"{fecha_min} a {fecha_max}")
        except:
            st.metric("Período", "Consultar datos")
    with col3:
        st.metric("Rango GEB", f"{df['GEB'].min():.0f} - {df['GEB'].max():.0f}")
    with col4:
        st.metric("Rango USD/COP", f"{df['USD_COP'].min():.0f} - {df['USD_COP'].max():.0f}")
    
    # Mostrar primeras filas
    st.subheader("📋 Vista previa de los datos")
    st.dataframe(df.head(15))
    
    # Descargar datos como CSV
    csv = df.to_csv()
    st.download_button(
        label="📥 Descargar datos como CSV",
        data=csv,
        file_name="geb_datos.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Gráfico de evolución temporal
    st.subheader("📈 Evolución Temporal de Variables")
    df_reset = df.reset_index()
    # Renombrar la columna de índice si no es 'Fecha'
    if 'Date' in df_reset.columns:
        df_reset.rename(columns={'Date': 'Fecha'}, inplace=True)
    elif df_reset.columns[0] not in ['Fecha', 'fecha']:
        df_reset.rename(columns={df_reset.columns[0]: 'Fecha'}, inplace=True)
    
    fig_time = px.line(df_reset, x="Fecha", y=["GEB", "USD_COP", "Brent"],
                       title="Series de Tiempo: GEB, USD/COP y Brent",
                       labels={"Fecha": "Fecha", "value": "Valor"},
                       markers=False)
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Validar columnas
    required_columns = ['GEB', 'USD_COP', 'Brent']
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        st.error(f"❌ Columnas faltantes: {missing}")
        st.warning(f"Columnas disponibles: {list(df.columns)}")
    else:
        # Calcular correlaciones
        corr_usd = df['GEB'].corr(df['USD_COP'])
        corr_brent = df['GEB'].corr(df['Brent'])
        
        st.subheader("📊 Coeficientes de Correlación de Pearson")
        col1, col2 = st.columns(2)
        col1.metric("Correlación: GEB vs USD/COP", f"{corr_usd:.3f}")
        col2.metric("Correlación: GEB vs Petróleo Brent", f"{corr_brent:.3f}")
        
        st.markdown("---")
        st.subheader("📈 Visualización de Relaciones")
        
        # Gráficos interactivos
        tab1, tab2 = st.tabs(["GEB vs USD/COP", "GEB vs Petróleo Brent"])
        
        with tab1:
            df_scatter = df.reset_index()
            if 'Date' in df_scatter.columns:
                df_scatter.rename(columns={'Date': 'Fecha'}, inplace=True)
            
            fig1 = px.scatter(
                df_scatter, x="USD_COP", y="GEB", trendline="ols",
                title="Dispersión: Precio Acción GEB vs Tasa de Cambio USD/COP",
                labels={"USD_COP": "Tasa de Cambio (COP)", "GEB": "Precio Acción GEB (COP)"},
                color_discrete_sequence=["#1f77b4"]
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.caption(f"Correlación: {corr_usd:.3f} - {'Positiva' if corr_usd > 0 else 'Negativa'}")
        
        with tab2:
            df_scatter = df.reset_index()
            if 'Date' in df_scatter.columns:
                df_scatter.rename(columns={'Date': 'Fecha'}, inplace=True)
            
            fig2 = px.scatter(
                df_scatter, x="Brent", y="GEB", trendline="ols",
                title="Dispersión: Precio Acción GEB vs Precio Petróleo Brent",
                labels={"Brent": "Petróleo Brent (USD)", "GEB": "Precio Acción GEB (COP)"},
                color_discrete_sequence=["#ff7f0e"]
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(f"Correlación: {corr_brent:.3f} - {'Positiva' if corr_brent > 0 else 'Negativa'}")
else:
    st.info("👈 Selecciona una opción en la barra izquierda para cargar datos y visualizar los gráficos")

# IMPORTANTE: Para correr el dashboard, ejecuta en la terminal: streamlit run GEB-Regresion.py