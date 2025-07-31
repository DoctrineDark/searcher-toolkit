from .Controller import Controller
from flask import current_app, request
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer
import threading
import logging

_model_cache = {}
_tokenizer_cache = {}
_cache_lock = threading.Lock()

class Embedding(Controller):
    @classmethod
    def vectorize(cls):
        try:
            data = request.get_json() or {}
            model_name = data.get("model", "BAAI/bge-m3")
            content = data.get("content")
            
            if not content:
                return {"success": False, "error": "Content is required"}, 400
            
            current_app.logger.info(f"Processing content: {content[:200]}...")
            
            with _cache_lock:
                #
                if model_name not in _tokenizer_cache:
                    current_app.logger.info(f"Initializing tokenizer for {model_name}...")
                    _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
                        model_name,
                        legacy=False,
                        use_fast=True
                    )
                tokenizer = _tokenizer_cache[model_name]
                
                #
                if model_name not in _model_cache:
                    current_app.logger.info(f"Loading model {model_name}...")
                    _model_cache[model_name] = BGEM3FlagModel(
                        model_name,
                        use_fp16=False,
                        device='cpu',
                        cache_folder="/app/models"  # Явное указание пути
                    )
                model = _model_cache[model_name]

            #
            tokens = tokenizer(
                content,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )
            token_count = tokens.input_ids.shape[1]
            
            #
            encoded = model.encode(
                [content],
                max_length=min(token_count, 512)
            )
            
            embedding_vector = encoded['dense_vecs'][0].tolist()
            vector_length = sum(x**2 for x in embedding_vector) ** 0.5
            is_normalized = abs(vector_length - 1.0) < 0.001
            
            return {
                'success': True,
                'embedding': embedding_vector,
                'dimension': model.model.config.hidden_size,
                'token_count': token_count,
                'is_normalized': is_normalized,
                'vector_length': float(vector_length)
            }, 200
            
        except Exception as e:
            current_app.logger.error(f"Critical error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'advice': 'Try reducing input length or restart service'
            }, 500