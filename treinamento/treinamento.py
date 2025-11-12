# Salve este como treinamento.py
import pandas as pd
from supabase import create_client, Client
from prophet import Prophet
import pickle # Para salvar o modelo
import logging
import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env para o ambiente
load_dotenv()

# --- Configurações ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
DEVICE_ID_GOLDEN = "sensor_ouro" # Sensor usado como "padrão ouro"
MODEL_FILE_NAME = "dashboard/modelo_normal.pkl"

# Silenciar logs do Prophet
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

print("--- Iniciando Treinamento do Modelo Padrão ---")

# --- 1. Conexão ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Conexão com Supabase estabelecida.")
except Exception as e:
    print(f"❌ Erro ao conectar: {e}")
    exit()

# --- 2. Carregamento de TODOS os dados do sensor "golden" ---
print(f"Buscando dados históricos para o sensor '{DEVICE_ID_GOLDEN}'...")
try:
    # Atenção: Em um cenário real, você pode precisar paginar os resultados
    # se tiver milhões de linhas. Para o PI, .limit(50000) deve bastar.
    response = supabase.table('sensores').select("*") \
                                       .eq('device_id', DEVICE_ID_GOLDEN) \
                                       .order('created_at', desc=False) \
                                       .limit(50000) \
                                       .execute()
    
    if not response.data:
        print(f"❌ Nenhum dado encontrado para '{DEVICE_ID_GOLDEN}'.")
        exit()
        
    df = pd.DataFrame(response.data)
    print(f"✅ {len(df)} registros baixados.")

except Exception as e:
    print(f"❌ Erro ao buscar dados: {e}")
    exit()

# --- 3. Preparação dos Dados ---
print("Preparando dados para o Prophet...")
try:
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['temperatura'] = pd.to_numeric(df['temperatura'])
    
    df_prophet = df.rename(
        columns={'created_at': 'ds', 'temperatura': 'y'}
    )
    # O Prophet exige que 'ds' não tenha fuso horário (seja "naive")
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    df_prophet = df_prophet[['ds', 'y']]
    
    if len(df_prophet) < 10: # Prophet precisa de um mínimo de dados
        print("❌ Dados insuficientes para treinamento.")
        exit()
        
    print("✅ Dados prontos.")

except Exception as e:
    print(f"❌ Erro ao preparar dados: {e}")
    exit()

# --- 4. Treinamento do Modelo ---
print("Iniciando treinamento... (Isso pode levar alguns minutos)")
try:
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=False
    )
    model.fit(df_prophet)
    print("✅ Modelo treinado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao treinar o modelo: {e}")
    exit()

# --- 5. Salvando o Modelo ---
print(f"Salvando modelo em '{MODEL_FILE_NAME}'...")
try:
    with open(MODEL_FILE_NAME, 'wb') as f:
        pickle.dump(model, f)
    print("✅ Modelo salvo com sucesso!")
except Exception as e:
    print(f"❌ Erro ao salvar o modelo: {e}")

print("\n--- Processo de Treinamento Concluído ---")