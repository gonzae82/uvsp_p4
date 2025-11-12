import requests
import json
import time
import os

# --- 1. PREENCHA SUAS CREDENCIAIS AQUI ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
# ---------------------------------------------

# O nome da sua tabela. Verifique se está correto.
TABLE_NAME = "sensores"

# Monta o endpoint
SUPABASE_ENDPOINT = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"

# Monta os headers (cabeçalhos)
SUPABASE_HEADERS = {
    "Content-Type": "application/json",
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}"
}
# --- Monta o Payload de Teste ---
payload = {
    "temperatura": 99.9,
    "umidade": 99.9,
    "device_id": f"teste_conexao_{int(time.time())}"  # ID único para o teste
}

print(f"--- Testando Conexão com Supabase ---")
print(f"Endpoint: {SUPABASE_ENDPOINT}")
print(f"Payload: {json.dumps(payload)}")
print("---------------------------------------")

try:
    response = requests.post(
        SUPABASE_ENDPOINT,
        headers=SUPABASE_HEADERS,
        data=json.dumps(payload)
    )

    # --- Análise da Resposta ---
    print(f"Status da Resposta: {response.status_code}")
    print("\nConteúdo da Resposta (Body):")
    print(response.text)
    
    print("\n--- Diagnóstico ---")
    if response.status_code == 201:
        print("✅ SUCESSO! Conexão validada e dados inseridos.")
        print("Vá ao painel do Supabase e procure pelo registro com 'device_id' iniciando em 'teste_conexao_'.")
    
    elif response.status_code == 401:
        print("❌ FALHA (401 - Unauthorized): Sua SUPABASE_KEY está errada ou inválida.")
    
    elif response.status_code == 404:
        print(f"❌ FALHA (404 - Not Found): O Endpoint está errado.")
        print(f"Verifique se o seu SUPABASE_URL ('{SUPABASE_URL}') e o nome da tabela ('{TABLE_NAME}') estão corretos.")
    
    elif response.status_code == 400:
        print("❌ FALHA (400 - Bad Request): O Supabase encontrou um problema com os dados.")
        print("Causa provável: Os nomes das colunas no 'payload' (temperatura, umidade, device_id) não batem com os da sua tabela no Supabase.")
        print("Ou, a coluna 'device_id' ainda não foi criada na sua tabela 'sensores'.")

    else:
        print(f"❌ FALHA (Código {response.status_code}): Ocorreu um erro desconhecido.")


except requests.exceptions.RequestException as e:
    print("\n--- Diagnóstico ---")
    print(f"❌ FALHA CRÍTICA (Erro de Rede): Não foi possível nem mesmo se conectar à URL.")
    print(f"Causa provável: O SUPABASE_URL ('{SUPABASE_URL}') está escrito errado ou você está sem internet.")
    print(f"Erro: {e}")