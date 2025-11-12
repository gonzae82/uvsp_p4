# Use uma imagem Python leve
FROM python:3.10-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia e instala os requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o script do simulador para o contêiner
# (O docker-compose.yml vai sobrepor isso com o 'volume',
# mas é uma boa prática para builds de produção)
COPY ./app/simulador.py .

# O comando que será executado quando o contêiner iniciar
CMD ["python", "simulador.py"]