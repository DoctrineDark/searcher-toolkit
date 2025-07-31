from flask import Flask
from werkzeug.serving import make_server
import logging
			 
import os
import psutil		   
																					  
from transformers import AutoTokenizer
from FlagEmbedding import BGEM3FlagModel

# ==== Конфиг offload ====
MODEL_NAME = "BAAI/bge-m3"
CACHE_FOLDER = "/app/models"
OFFLOAD_FOLDER = "/app/offload"

os.makedirs(CACHE_FOLDER, exist_ok=True)
os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

# ==== Flask ====
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== Кэш модели и токенизатора ====
_model_cache = {}
_tokenizer_cache = {}

def log_memory(stage):
																									
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory {stage}: {mem_mb:.2f} MB")

# ==== Предзагрузка ====
def preload_model():
							  
    log_memory("before preload")

										  
    logger.info(f"Preloading tokenizer for {MODEL_NAME}...")
    _tokenizer_cache[MODEL_NAME] = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        legacy=False,
        use_fast=True
    )
    log_memory("after tokenizer")

									  
    logger.info(f"Preloading model {MODEL_NAME} with offload...")
    _model_cache[MODEL_NAME] = BGEM3FlagModel(
        MODEL_NAME,
        device='cpu',
        use_fp16=True,  # половинная точность → меньше RAM
        cache_folder=CACHE_FOLDER,
        model_kwargs={
            "device_map": "auto",
            "offload_folder": OFFLOAD_FOLDER,
            "low_cpu_mem_usage": True
        }
    )
    log_memory("after model")

preload_model()

# ==== Роуты ====
from resources.Home import Home
from resources.Embedding import Embedding

app.add_url_rule('/', view_func=Home.index, methods=['GET'])
app.add_url_rule('/embedding', view_func=Embedding.vectorize, methods=['POST'])

# ==== Старт сервера ====
if __name__ == '__main__':
    logger.info("Starting server...")
    log_memory("before serve_forever")

    server = make_server(
        host='0.0.0.0',
        port=5000,
        app=app,
        threaded=True,
        processes=1
    )
 
    try:
        logger.info("Server running on http://0.0.0.0:5000")
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
