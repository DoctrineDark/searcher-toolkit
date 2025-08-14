FROM python:3.10

WORKDIR /app

RUN mkdir -p ${HF_HOME} && chmod -R 777 ${HF_HOME}

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

#RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-m3', local_files_only=False)"

COPY app/ ./app/

EXPOSE 5000

CMD ["python", "-u", "app/main.py"]
