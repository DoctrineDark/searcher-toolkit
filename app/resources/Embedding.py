from flask import current_app, request
from .Controller import Controller
import logging
import threading

# Импортируем предзагруженные объекты из main.py
from __main__ import _model_cache, _tokenizer_cache, MODEL_NAME

_cache_lock = threading.Lock()

class Embedding(Controller):
    @classmethod
    def vectorize(cls):
        try:
            data = request.get_json() or {}
            model_name = data.get("model", MODEL_NAME)
            content = data.get("content")

            if not content:
                return {"success": False, "error": "Content is required"}, 400

            current_app.logger.info(f"Processing content: {content[:200]}...")

            # Берём модель и токенизатор из кэша
            with _cache_lock:
                tokenizer = _tokenizer_cache[model_name]
                model = _model_cache[model_name]

            # Токенизация
            tokens = tokenizer(
                content,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )
            token_count = tokens.input_ids.shape[1]

            # Векторизация
            encoded = model.encode(
                [content],
                max_length=min(token_count, 512)
            )

            return {
                'success': True,
                'embedding': encoded['dense_vecs'][0].tolist(),
                'dimension': model.model.config.hidden_size,
                'token_count': token_count
            }, 200

        except Exception as e:
            current_app.logger.error(f"Critical error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'advice': 'Try reducing input length or restart service'
            }, 500

