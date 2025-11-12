# 1. Imagem Base
FROM python:3.10-slim

# 2. Define o diretório de trabalho
WORKDIR /app

# 3. Copia o arquivo de dependências
COPY requirements.txt .

# 4. Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia o script do dashboard
COPY app.py .
COPY modelo_normal.pkl .

# 6. Expõe a porta do Streamlit
EXPOSE 9501

# 7. Comando para iniciar o Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=9501", "--server.address=0.0.0.0"]