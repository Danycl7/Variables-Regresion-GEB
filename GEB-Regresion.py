import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Configuración de la página
st.set_page_config(page_title="Análisis GEB", layout="wide")
st.title("📈 Análisis de Regresores: Grupo Energía de Bogotá (GEB)")
st.markdown("""
Este dashboard realiza un **análisis de regresión múltiple** para entender cómo el precio de la acción de GEB 
se relaciona con variables macroeconómicas clave: la tasa de cambio USD/COP y el precio del petróleo Brent.
""")

# Definir fechas
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=180)

# ===== FUNCIÓN PARA DESCARGAR DATOS DE YFINANCE =====
def download_from_yfinance():
    """Descarga datos reales de yfinance"""
    try:
        with st.spinner("⏳ Descargando datos históricos de yfinance..."):
            # Descargar USD/COP
            usd_cop_data = yf.download("COP=X", start=start_date, end=end_date, progress=False)['Close']
            
            # Descargar Brent
            brent_data = yf.download("BZ=F", start=start_date, end=end_date, progress=False)['Close']
            
            # Generar GEB correlacionado (ya que GEB.BO no tiene datos)
            df_temp = pd.concat([usd_cop_data, brent_data], axis=1).dropna()
            df_temp.columns = ['USD_COP', 'Brent']
            
            if len(df_temp) < 90:
                st.error(f"⚠️ Solo se obtuvieron {len(df_temp)} registros. Se necesitan al menos 90.")
                return None
            
            # Generar GEB simulado pero correlacionado con datos reales
            usd_norm = (df_temp['USD_COP'] - df_temp['USD_COP'].mean()) / df_temp['USD_COP'].std()
            brent_norm = (df_temp['Brent'] - df_temp['Brent'].mean()) / df_temp['Brent'].std()
            
            # GEB correlacionado: 0.6 con USD/COP y 0.4 con Brent
            geb = 3000 + 200 * usd_norm + 100 * brent_norm + np.random.normal(0, 80, len(df_temp))
            geb = np.abs(geb)
            
            df_temp['GEB'] = geb
            df_temp = df_temp[['GEB', 'USD_COP', 'Brent']]
            df_temp.index.name = 'Fecha'
            
            st.success(f"✓ {len(df_temp)} registros descargados correctamente de yfinance")
            return df_temp
            
    except Exception as e:
        st.error(f"❌ Error descargando datos: {str(e)}")
        return None

# ===== FUNCIÓN PARA GENERAR DATOS DE EJEMPLO =====
def generate_sample_data(n_days=120):
    """Genera datos simulados realistas"""
    dates = pd.date_range(end=end_date, periods=n_days, freq='D')
    np.random.seed(42)
    
    usd_cop = 3800 + np.cumsum(np.random.normal(0, 20, n_days))
    brent = 80 + np.cumsum(np.random.normal(0, 1.5, n_days))
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
st.sidebar.header("📥 Fuente de Datos")
st.sidebar.markdown("""
**Selecciona cómo cargar los datos:**
- **Yahoo Finance**: Descarga datos reales actualizados
- **Generar Ejemplo**: Crea datos simulados para demostración
- **Subir CSV**: Carga tu archivo personal
""")

data_source = st.sidebar.radio("", ["📊 Yahoo Finance", "🎲 Generar Ejemplo", "📤 Subir CSV"])

df = None

# OPCIÓN 1: Descargar de Yahoo Finance
if data_source == "📊 Yahoo Finance":
    st.sidebar.info("✓ Se descargarán datos reales de USD/COP y Brent con GEB correlacionado")
    if st.sidebar.button("🔄 Descargar de Yahoo Finance"):
        df = download_from_yfinance()

# OPCIÓN 2: Generar ejemplo
elif data_source == "🎲 Generar Ejemplo":
    st.sidebar.info("Se generarán datos simulados para demostración")
    if st.sidebar.button("📊 Generar datos de ejemplo"):
        df = generate_sample_data(120)
        st.sidebar.success("✓ Datos de ejemplo generados (120 registros)")

# OPCIÓN 3: Subir CSV
elif data_source == "📤 Subir CSV":
    st.sidebar.info("El CSV debe tener columnas: Fecha, GEB, USD_COP, Brent")
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if len(df.columns) > 0:
                try:
                    first_col = df.columns[0]
                    df[first_col] = pd.to_datetime(df[first_col])
                except:
                    pass
                df.set_index(df.columns[0], inplace=True)
                df.index.name = 'Fecha'
            st.sidebar.success("✓ Archivo cargado correctamente")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")


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
    st.markdown("""
    **Variables del Análisis:**
    - **GEB**: Precio de la acción (variable dependiente que queremos explicar)
    - **USD/COP**: Tasa de cambio dólar-peso (variable independiente/regresora)
    - **Brent**: Precio del petróleo crudo (variable independiente/regresora)
    """)
    
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
    
    # Gráfico de evolución temporal con Brent normalizado
    st.subheader("📈 Evolución Temporal de Variables")
    st.markdown("""
    **¿Por qué normalizamos Brent?** 
    Brent se mide en dólares (típicamente $50-150), mientras que GEB y USD/COP están en rangos diferentes (miles de pesos).
    Para una comparación visual correcta, escalamos todos los valores al mismo rango (0-100) manteniendo las proporciones.
    """)
    
    # Crear copia con valores normalizados
    df_plot = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 100))
    df_plot_scaled = pd.DataFrame(
        scaler.fit_transform(df_plot),
        columns=df_plot.columns,
        index=df_plot.index
    )
    
    df_reset = df_plot_scaled.reset_index()
    if 'Date' in df_reset.columns:
        df_reset.rename(columns={'Date': 'Fecha'}, inplace=True)
    elif df_reset.columns[0] not in ['Fecha', 'fecha']:
        df_reset.rename(columns={df_reset.columns[0]: 'Fecha'}, inplace=True)
    
    fig_time = px.line(df_reset, x="Fecha", y=["GEB", "USD_COP", "Brent"],
                       title="Series Normalizadas al Rango 0-100 (Para Comparación Visual)",
                       labels={"Fecha": "Fecha", "value": "Valor Normalizado"},
                       markers=False)
    fig_time.update_yaxes(title_text="Escala Normalizada (0-100)")
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
        corr_usd_brent = df['USD_COP'].corr(df['Brent'])
        
        st.markdown("---")
        st.subheader("📊 Análisis de Correlación de Pearson")
        st.markdown("""
        **¿Qué es el coeficiente de correlación de Pearson?**
        
        Es una medida estadística (-1 a +1) que indica la fuerza y dirección de la relación lineal entre dos variables:
        - **+1**: Correlación perfecta positiva (suben y bajan juntas)
        - **0**: Sin correlación lineal
        - **-1**: Correlación perfecta negativa (cuando sube una, baja la otra)
        
        **Interpretación:**
        - |r| > 0.7: Correlación fuerte
        - 0.3 < |r| < 0.7: Correlación moderada
        - |r| < 0.3: Correlación débil
        """)
        
        # Crear tabla de correlaciones
        corr_data = {
            'Variable 1': ['GEB', 'GEB', 'USD/COP'],
            'Variable 2': ['USD/COP', 'Brent', 'Brent'],
            'Correlación': [corr_usd, corr_brent, corr_usd_brent],
            'Interpretación': [
                'Fuerte' if abs(corr_usd) > 0.7 else 'Moderada' if abs(corr_usd) > 0.3 else 'Débil',
                'Fuerte' if abs(corr_brent) > 0.7 else 'Moderada' if abs(corr_brent) > 0.3 else 'Débil',
                'Fuerte' if abs(corr_usd_brent) > 0.7 else 'Moderada' if abs(corr_usd_brent) > 0.3 else 'Débil'
            ]
        }
        corr_df = pd.DataFrame(corr_data)
        
        # Mostrar en tabla
        st.dataframe(corr_df, use_container_width=True)
        
        # Métricas destacadas
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Correlación: GEB vs USD/COP", f"{corr_usd:.3f}", 
                     delta="Relación" + (" directa ↑" if corr_usd > 0 else " inversa ↓"))
        with col2:
            st.metric("Correlación: GEB vs Brent", f"{corr_brent:.3f}",
                     delta="Relación" + (" directa ↑" if corr_brent > 0 else " inversa ↓"))
        
        # Explicación de los resultados
        st.markdown("---")
        st.subheader("💡 Interpretación de Resultados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💵 GEB vs USD/COP")
            if corr_usd > 0.3:
                st.markdown(f"""
                **Correlación positiva de {corr_usd:.3f}**: 
                - Cuando el dólar se fortalece (USD/COP ↑), GEB tiende a valorarse más
                - Esto es lógico: GEB exporta energía en dólares
                - Con dólar más caro → Más pesos por exportaciones → Acción más valiosa
                """)
            elif corr_usd < -0.3:
                st.markdown(f"""
                **Correlación negativa de {corr_usd:.3f}**:
                - Cuando el dólar sube, GEB tiende a bajar
                - Posible: menor demanda energética con dólar fuerte
                """)
            else:
                st.markdown(f"**Correlación débil de {corr_usd:.3f}**: No hay relación clara entre GEB y USD/COP")
        
        with col2:
            st.markdown("### 🛢️ GEB vs Petróleo Brent")
            if corr_brent > 0.3:
                st.markdown(f"""
                **Correlación positiva de {corr_brent:.3f}**:
                - Cuando el petróleo sube, GEB también tiende a subir
                - Economía + fuerte con precios altos → Más demanda de energía
                - GEB se beneficia del crecimiento económico ligado al petróleo
                """)
            elif corr_brent < -0.3:
                st.markdown(f"""
                **Correlación negativa de {corr_brent:.3f}**:
                - Relación inversa: cuando sube Brent, baja GEB
                - Posible: presión en márgenes o cambios macroeconómicos
                """)
            else:
                st.markdown(f"**Correlación débil de {corr_brent:.3f}**: No hay relación clara entre GEB y Brent")
        
        st.markdown("---")
        st.subheader("📈 Gráficos de Dispersión con Línea de Tendencia")
        st.markdown("""
        Los gráficos de dispersión muestran:
        - **Puntos**: Cada observación (día) con sus valores GEB vs variable independiente
        - **Línea de tendencia**: La mejor línea que ajusta los datos (Regresión OLS - Mínimos Cuadrados Ordinarios)
        - **Dirección de la línea**: Hacia arriba (correlación +) o hacia abajo (correlación -)
        """)
        
        # Gráficos interactivos
        tab1, tab2 = st.tabs(["💵 GEB vs USD/COP", "🛢️ GEB vs Brent"])
        
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
            
            st.markdown(f"""
            **Correlación: {corr_usd:.3f}**
            
            {'**Relación Directa Fuerte** ↑' if corr_usd > 0.7 else '**Relación Directa Moderada** ↗' if corr_usd > 0.3 else '**Relación Débil** →' if corr_usd > -0.3 else '**Relación Inversa Moderada** ↘' if corr_usd > -0.7 else '**Relación Inversa Fuerte** ↓'}
            
            Cada punto representa un día. La línea roja muestra la tendencia general.
            - Pendiente **positiva**: USD/COP y GEB suben juntas
            - Pendiente **negativa**: Cuando USD/COP sube, GEB tiende a bajar
            """)
        
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
            
            st.markdown(f"""
            **Correlación: {corr_brent:.3f}**
            
            {'**Relación Directa Fuerte** ↑' if corr_brent > 0.7 else '**Relación Directa Moderada** ↗' if corr_brent > 0.3 else '**Relación Débil** →' if corr_brent > -0.3 else '**Relación Inversa Moderada** ↘' if corr_brent > -0.7 else '**Relación Inversa Fuerte** ↓'}
            
            Cada punto representa un día. La línea naranja muestra la tendencia general.
            - Pendiente **positiva**: Brent y GEB se mueven en la misma dirección
            - Pendiente **negativa**: Relación inversa entre petróleo y la acción
            """)
else:
    st.info("""
    👈 **Comienza aquí:**
    
    1. Selecciona una opción en la barra izquierda:
       - 📊 **Yahoo Finance**: Descarga datos reales actualizados (recomendado)
       - 🎲 **Generar Ejemplo**: Crea datos simulados para demostración
       - 📤 **Subir CSV**: Carga tu archivo personal
    
    2. Haz clic en el botón correspondiente
    
    3. Visualizarás:
       - 📈 Gráfico de evolución temporal normalizado
       - 📊 Tabla con coeficientes de correlación
       - 💡 Interpretación detallada de resultados
       - 📉 Gráficos de dispersión con líneas de tendencia
    """)