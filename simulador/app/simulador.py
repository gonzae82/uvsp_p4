import os
import requests
import json
import time
import threading
import numpy as np
import random

# --- 1. Carregar Configuração do Ambiente (do docker-compose.yml) ---
print("Carregando variáveis de ambiente...")
# Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_ENDPOINT = f"{SUPABASE_URL}/rest/v1/sensores"
SUPABASE_HEADERS = {
    "Content-Type": "application/json",
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}"
}

# Simulação
NUM_NORMAL_DEVICES = int(os.environ.get("NUM_NORMAL_DEVICES", 5))
NUM_ANOMALY_DEVICES = int(os.environ.get("NUM_ANOMALY_DEVICES", 2))
SEND_INTERVAL = int(os.environ.get("SEND_INTERVAL", 60))

# Parâmetros Normais
TEMP_NORMAL_MEDIA = float(os.environ.get("TEMP_NORMAL_MEDIA", 22.0))
TEMP_NORMAL_STD = float(os.environ.get("TEMP_NORMAL_STD", 0.5))
HUM_NORMAL_MEDIA = float(os.environ.get("HUM_NORMAL_MEDIA", 45.0))
HUM_NORMAL_STD = float(os.environ.get("HUM_NORMAL_STD", 2.0))

# Parâmetros de Anomalia
TEMP_ANOMALY_MEDIA = float(os.environ.get("TEMP_ANOMALY_MEDIA", 28.0))
TEMP_ANOMALY_STD = float(os.environ.get("TEMP_ANOMALY_STD", 1.5))
HUM_ANOMALY_MEDIA = float(os.environ.get("HUM_ANOMALY_MEDIA", 40.0))
HUM_ANOMALY_STD = float(os.environ.get("HUM_ANOMALY_STD", 1.0))


# --- 2. Lógica do Simulador ---

class DeviceSimulator(threading.Thread):
    """
    Representa um dispositivo IoT (sensor DHT22) virtual.
    Cada dispositivo roda em sua própria thread.
    """
    def __init__(self, device_id, is_anomaly=False):
        super().__init__()
        self.device_id = device_id
        self.is_anomaly = is_anomaly
        self.daemon = True  # Permite que o programa feche mesmo se as threads estiverem rodando

    def generate_data(self):
        """ Gera dados de telemetria baseados no perfil do dispositivo. """
        
        #!# ALTERAÇÃO: Adicionada lógica específica para o "sensor_ouro"
        # Este 'if' tem prioridade sobre os outros
        if self.device_id == "sensor_ouro":
            # Simula condições perfeitas: ar-condicionado ideal.
            # Média de 21°C com desvio padrão *muito baixo* (0.1)
            temp = np.random.normal(21.0, 0.1) 
            # Média de 45% de umidade com desvio padrão *baixo* (0.5)
            hum = np.random.normal(45.0, 0.5)
        
        elif self.is_anomaly: #!# ALTERAÇÃO: 'if' mudou para 'elif'
            # Simula um ambiente com falha no ar-condicionado
            temp = np.random.normal(TEMP_ANOMALY_MEDIA, TEMP_ANOMALY_STD)
            hum = np.random.normal(HUM_ANOMALY_MEDIA, HUM_ANOMALY_STD)
            
            # Adiciona uma chance de "pico" extremo (ex: sensor com defeito)
            if random.random() < 0.1: # 10% de chance
                temp += random.uniform(5, 10) 
                
        else:
            # Simula um ambiente normal e controlado
            temp = np.random.normal(TEMP_NORMAL_MEDIA, TEMP_NORMAL_STD)
            hum = np.random.normal(HUM_NORMAL_MEDIA, HUM_NORMAL_STD)

        # Arredonda para 2 casas decimais, como o ESP32 faria
        return round(temp, 2), round(hum, 2)

    def send_to_supabase(self, temperatura, umidade):
        """
        Envia os dados para o endpoint do Supabase.
        Imita exatamente a requisição POST do código ESP32.
        """
        try:
            # IMPORTANTE: Adicionamos o 'device_id' ao payload.
            # Isso é crucial para o ML!
            payload = {
                "temperatura": temperatura,
                "umidade": umidade,
                "device_id": self.device_id  # <-- VEJA A OBSERVAÇÃO ABAIXO
            }
            
            response = requests.post(
                SUPABASE_ENDPOINT,
                headers=SUPABASE_HEADERS,
                data=json.dumps(payload)
            )
            
            if response.status_code == 201: # 201 Created
                print(f"[Device {self.device_id}]: Dados enviados com sucesso ({temperatura}°C, {umidade}%)")
            else:
                print(f"[Device {self.device_id}]: Erro ao enviar dados. Status: {response.status_code}")
                print(f"[Device {self.device_id}]: Resposta: {response.text}")

        except Exception as e:
            print(f"[Device {self.device_id}]: Exceção ao enviar dados: {e}")

    def run(self):
        """ O loop principal do dispositivo (como o 'loop()' do Arduino). """
        print(f"Iniciando dispositivo: {self.device_id} (Anomalia: {self.is_anomaly})")
        while True:
            temp, hum = self.generate_data()
            self.send_to_supabase(temp, hum)
            
            # Adiciona uma pequena variação no intervalo de envio
            # para não sobrecarregar o banco com requisições simultâneas
            sleep_time = SEND_INTERVAL + random.uniform(-5, 5)
            time.sleep(sleep_time)

# --- 3. Ponto de Entrada Principal ---

def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Erro: Variáveis de ambiente SUPABASE_URL e SUPABASE_KEY não definidas.")
        print("Por favor, configure-as no arquivo docker-compose.yml")
        return

    print("--- Iniciando Simulador de Data Center ---")
    print(f"Endpoint: {SUPABASE_ENDPOINT}")
    
    devices = []
    
    #!# ALTERAÇÃO: Criar e adicionar o sensor_ouro primeiro
    print("Criando Sensor Ouro (Padrão)...")
    gold_dev = DeviceSimulator(device_id="sensor_ouro", is_anomaly=False)
    devices.append(gold_dev)
    
    # Criar dispositivos normais
    for i in range(NUM_NORMAL_DEVICES):
        device_id = f"sensor_normal_{i+1:02d}"
        dev = DeviceSimulator(device_id, is_anomaly=False)
        devices.append(dev)

    # Criar dispositivos com anomalia
    for i in range(NUM_ANOMALY_DEVICES):
        device_id = f"sensor_anomalia_{i+1:02d}"
        dev = DeviceSimulator(device_id, is_anomaly=True)
        devices.append(dev)

    # Iniciar todas as threads
    for dev in devices:
        dev.start()

    # Manter o script principal rodando
    try:
        while True:
            time.sleep(3600) # Apenas "durma" por uma hora de cada vez
    except KeyboardInterrupt:
        print("\n--- Simulador encerrado ---")

if __name__ == "__main__":
    main()