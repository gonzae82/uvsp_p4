# Salve este como app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
import pytz
from datetime import timedelta
import logging
import pickle
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import os

# --- Configuração da Página ---
st.set_page_config(
    page_title="Monitoramento de Temperatura dos Sensores",
    page_icon="🖥️",
    layout="wide"
)

# --- Constantes Globais ---
# Utilizar no Docker
#MODEL_FILE_NAME = "dashboard/modelo_normal.pkl"

# Utilizar no desenvolvimento local
MODEL_FILE_NAME = "modelo_normal.pkl"

FUSO_SP = pytz.timezone("America/Sao_Paulo")

# --- Silenciar Logs ---
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# --- 1. FUNÇÕES DE CARREGAMENTO E CACHE ---

@st.cache_resource
def init_connection() -> Client:
    """
    Inicializa a conexão com o Supabase.
    Usa @st.cache_resource para rodar apenas uma vez.
    """
    try:
       # url = st.secrets["SUPABASE_URL"]
       # key = st.secrets["SUPABASE_KEY"]
        
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        
        
        return create_client(url, key)
    except KeyError:
        st.error("Erro: Credenciais 'SUPABASE_URL' ou 'SUPABASE_KEY' não encontradas.")
        st.info("Por favor, crie o arquivo .streamlit/secrets.toml com suas credenciais.")
        st.stop()

@st.cache_resource
def load_prophet_model(filename: str) -> Prophet:
    """
    Carrega o modelo Prophet pré-treinado do arquivo .pkl.
    Usa @st.cache_resource para carregar o modelo na memória apenas uma vez.
    """
    if not os.path.exists(filename):
        st.error(f"Arquivo do modelo '{filename}' não encontrado.")
        st.info(f"Por favor, execute treinamento primeiro.")
        st.stop()
    
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.stop()

@st.cache_data(ttl=60)
def load_latest_readings(_supabase: Client) -> pd.DataFrame:
    """
    Carrega a leitura MAIS RECENTE de CADA sensor para a tabela principal.
    Usa @st.cache_data com TTL de 60s para atualizar os dados periodicamente.
    """
    try:
        response = _supabase.table('sensores').select("*") \
                                            .order('created_at', desc=True) \
                                            .limit(2000) \
                                            .execute()
        if not response.data: 
            return pd.DataFrame() # Retorna DF vazio se não houver dados

        df = pd.DataFrame(response.data)
        # Pega a última entrada de cada 'device_id'
        df_latest = df.sort_values('created_at').drop_duplicates('device_id', keep='last')
        
        # Converte o fuso horário para exibição correta
        df_latest['created_at'] = pd.to_datetime(df_latest['created_at'], utc=True)
        df_latest['created_at'] = df_latest['created_at'].dt.tz_convert(FUSO_SP)
        
        df_latest['temperatura'] = pd.to_numeric(df_latest['temperatura'])
        df_latest['umidade'] = pd.to_numeric(df_latest['umidade'])
        return df_latest.sort_values('device_id')
    
    except Exception as e:
        st.error(f"Erro ao carregar dados da visão geral: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_historical_data(_supabase: Client, device_id: str) -> pd.DataFrame:
    """
    Carrega os últimos 2000 registros históricos de UM sensor específico.
    Usa @st.cache_data com TTL de 60s.
    """
    try:
        response = _supabase.table('sensores').select("*") \
                                            .eq('device_id', device_id) \
                                            .order('created_at', desc=True) \
                                            .limit(2000) \
                                            .execute()
        if not response.data: 
            return pd.DataFrame()

        df = pd.DataFrame(response.data)
        
        # Prepara o DataFrame para gráficos de série temporal
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
        df['created_at'] = df['created_at'].dt.tz_convert(FUSO_SP)
        df = df.set_index('created_at').sort_index() # Ordena do mais antigo ao mais novo
        
        df['temperatura'] = pd.to_numeric(df['temperatura'])
        df['umidade'] = pd.to_numeric(df['umidade'])
        return df
    
    except Exception as e:
        st.error(f"Erro ao carregar dados históricos: {e}")
        return pd.DataFrame()

# --- 2. FUNÇÕES DE LÓGICA / INFERÊNCIA ---
@st.cache_data(ttl=60)
def get_overview_status(_model: Prophet, df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe os dados mais recentes e usa o modelo para prever o status.
    Esta função encapsula a lógica de inferência da página principal.
    
    CORREÇÃO: '_model' informa ao Streamlit para não "hashar" o modelo.
    """
    # 1. Prepara os timestamps para o modelo (precisam ser 'ds' e sem fuso)
    df_to_predict = pd.DataFrame()
    df_to_predict['ds'] = df_latest['created_at'].dt.tz_localize(None)

    # 2. Faz a previsão (inferência). Isso é RÁPIDO.
    #    Note que usamos '_model' aqui.
    forecast = _model.predict(df_to_predict)
    
    # 3. Compara o real (temperatura) com o previsto (yhat_lower, yhat_upper)
    df_latest['yhat_lower'] = forecast['yhat_lower'].values
    df_latest['yhat_upper'] = forecast['yhat_upper'].values
    
    df_latest['anomalia'] = (df_latest['temperatura'] < df_latest['yhat_lower']) | \
                            (df_latest['temperatura'] > df_latest['yhat_upper'])
    
    df_latest['status'] = df_latest['anomalia'].apply(
        lambda x: "⚠️ Anomalia" if x else "✅ Normal"
    )
    return df_latest

@st.cache_data(ttl=60)
def get_detail_predictions(_model: Prophet, df_historical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recebe o histórico de um sensor e gera a previsão completa para ele.
    Retorna o 'forecast' original (para componentes) e o 'df_comparison' (para o gráfico).
    
    CORREÇÃO: '_model' informa ao Streamlit para não "hashar" o modelo.
    """
    # 1. Prepara dados históricos para o Prophet
    df_to_predict = df_historical.reset_index()[['created_at']]
    df_to_predict = df_to_predict.rename(columns={'created_at': 'ds'})
    df_to_predict['ds'] = df_to_predict['ds'].dt.tz_localize(None)

    # 2. Faz a previsão sobre todo o histórico (RÁPIDO)
    #    Note que usamos '_model' aqui.
    forecast = _model.predict(df_to_predict)
    
    # 3. Prepara uma CÓPIA do forecast para juntar com os dados reais
    forecast_for_join = forecast.copy()
    # Devolve o fuso horário e define como índice para o 'join'
    forecast_for_join['ds'] = forecast_for_join['ds'].dt.tz_localize(FUSO_SP.zone) 
    forecast_for_join = forecast_for_join.set_index('ds')

    # 4. Junta os dados reais (df_historical) com a previsão (forecast_for_join)
    df_comparison = df_historical.join(forecast_for_join[['yhat', 'yhat_lower', 'yhat_upper']])
    
    # Preenche possíveis 'NaN' que podem surgir no 'join'
    df_comparison = df_comparison.ffill().bfill() 
    
    # 5. Define as anomalias
    df_comparison['anomalia'] = (df_comparison['temperatura'] < df_comparison['yhat_lower']) | \
                                (df_comparison['temperatura'] > df_comparison['yhat_upper'])
    
    return forecast, df_comparison


# --- 3. FUNÇÕES DE RENDERIZAÇÃO (UI) ---

def render_overview_page(supabase: Client, model: Prophet):
    """Renderiza a página principal (tabela de visão geral)."""
    st.title("🌡️ Dashboard de Sensores de Temperatura - DataCenter")
    
    st.subheader("Projeto Integrador IV - Univesp - 2025")

    if st.button("🔄 Recarregar Dados"):
        st.cache_data.clear()

    df_latest = load_latest_readings(supabase)

    if not df_latest.empty:
        # 1. Pega os dados e...
        # 2. ...envia para a função de lógica obter o status
        df_display = get_overview_status(model, df_latest.copy()) # Usa .copy() p/ segurança
        
        # 3. Prepara colunas para exibição
        df_display['detalhes_link'] = df_display['device_id'].apply(lambda id: f"/?device_id={id}")
        df_display['data'] = df_display['created_at'].dt.strftime('%d/%m/%Y')
        df_display['hora'] = df_display['created_at'].dt.strftime('%H:%M:%S')
        
        # 4. Renderiza a tabela
        st.dataframe(
            df_display,
            hide_index=True,
            #use_container_width=True,
            width='stretch',
            column_order=["status", "device_id", "temperatura", "umidade", "data", "hora", "detalhes_link"],
            column_config={
                "status": st.column_config.TextColumn("Status Preditivo"),
                "device_id": st.column_config.TextColumn("Dispositivo"),
                "temperatura": st.column_config.NumberColumn("🌡️ Temp. Atual (°C)", format="%.1f"),
                "umidade": st.column_config.NumberColumn("💧 Umidade Atual (%)", format="%.1f"),
                "data": st.column_config.TextColumn("📅 Data"),
                "hora": st.column_config.TextColumn("⏰ Hora"),
                "detalhes_link": st.column_config.LinkColumn("Analisar 🔎", display_text="Ver Detalhes"),
                # Oculta colunas de trabalho
                "created_at": None, "id": None, "anomalia": None, 
                "yhat_lower": None, "yhat_upper": None,
            }
        )
    else:
        st.info("Aguardando dados dos sensores...")

def render_detail_page(supabase: Client, model: Prophet, device_id: str):
    """Renderiza a página de análise profunda para um único sensor."""
    st.title(f"🌡️ {device_id}")
    st.markdown("⬅️ [Voltar](/)") # Link para voltar

    if st.button("🔄 Recarregar Dados"):
        st.cache_data.clear()
        
    df = load_historical_data(supabase, device_id)
    
    if not df.empty:
        # 1. Pega os dados históricos e...
        # 2. ...envia para a função de lógica obter as previsões
        forecast, df_comparison = get_detail_predictions(model, df)
        
        # --- 3. Renderiza Métricas (KPIs) ---
        latest_data = df.iloc[-1] # Pega a última linha dos dados REAIS
        latest_status = df_comparison.iloc[-1] # Pega a última linha da COMPARAÇÃO
        
        # Calcula delta da última hora
        one_hour_ago = latest_data.name - timedelta(hours=1)
        past_data = df[df.index <= one_hour_ago]
        delta_temp, delta_umid = 0, 0
        if not past_data.empty:
            delta_temp = latest_data['temperatura'] - past_data.iloc[-1]['temperatura']
            delta_umid = latest_data['umidade'] - past_data.iloc[-1]['umidade']

        # Define o status da métrica
        if latest_status['anomalia']:
            status_text = "Anomalia Detectada"
            status_delta = f"{latest_status['temperatura']:.1f}° (Fora do esperado)"
            delta_color = "inverse"
        else:
            status_text = "Operação Normal"
            status_delta = "Dentro do esperado"
            delta_color = "off"

        col1, col2, col3 = st.columns(3)
        col1.metric("🌡️ Temperatura Atual", f"{latest_data['temperatura']:.1f} °C", f"{delta_temp:+.1f} °C (1h)")
        col2.metric("💧 Umidade Atual", f"{latest_data['umidade']:.1f} %", f"{delta_umid:+.1f} % (1h)")
        col3.metric("🤖 Status Preditivo", status_text, status_delta, delta_color)
        
        st.markdown("---")

        # --- 4. Renderiza Gráfico Principal de Anomalia ---
        st.subheader("Análise Preditiva vs. Real")
        df_plot = df_comparison.reset_index() # Plotly precisa da data como coluna
        
        fig = px.line(df_plot, x='created_at', y='temperatura', title=f"Análise de Anomalia para {device_id}")
        fig.update_traces(name='Temp. Real') 
        
        # Adiciona bandas de previsão
        fig.add_scatter(x=df_plot['created_at'], y=df_plot['yhat_upper'], mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Banda Superior')
        fig.add_scatter(x=df_plot['created_at'], y=df_plot['yhat_lower'], mode='lines', line=dict(color='rgba(0,100,80,0.2)'), fill='tonexty', fillcolor='rgba(0,100,80,0.2)', name='Banda Inferior')
        
        # Adiciona pontos de anomalia (em vermelho)
        df_anomalias = df_comparison[df_comparison['anomalia']]
        if not df_anomalias.empty:
            fig.add_scatter(x=df_anomalias.index, y=df_anomalias['temperatura'], 
                            mode='markers', marker=dict(color='red', size=5), name='Anomalia')

        # Move a legenda para baixo
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

        # --- 5. Renderiza Gráfico de Componentes ---
        st.subheader("Componentes do Modelo 'Normal'")
        # Usa o 'forecast' original (com a coluna 'ds')
        fig_comp = plot_components_plotly(model, forecast)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Registros Recentes")
        st.dataframe(df.tail(100), use_container_width=True)
    else:
        st.info(f"Aguardando dados históricos para '{device_id}'...")

# --- 4. PONTO DE ENTRADA PRINCIPAL (ROTEADOR) ---
def main():
    """Função principal que carrega os recursos e roteia a página."""
    
    # Carrega recursos essenciais UMA VEZ
    supabase = init_connection()
    model = load_prophet_model(MODEL_FILE_NAME)
    
    # Lógica de roteamento
    query_params = st.query_params
    
    if "device_id" in query_params:
        # Se a URL tiver ?device_id=..., mostra a página de detalhes
        render_detail_page(supabase, model, query_params["device_id"])
    else:
        # Senão, mostra a página principal
        render_overview_page(supabase, model)

if __name__ == "__main__":
    main()