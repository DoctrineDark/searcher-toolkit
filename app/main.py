from flask import Flask
from werkzeug.serving import make_server
import logging
from resources.Home import Home
from resources.Embedding import Embedding

from resources.Controller import Controller

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_url_rule('/', view_func=Home.index, methods=['GET'])
app.add_url_rule('/embedding', view_func=Embedding.vectorize, methods=['POST'])

if __name__ == '__main__':
    logger.info("Starting server...")
    
    server = make_server(
        host='0.0.0.0',
        port=5000,
        app=app,
        threaded=True,
        processes=1
    )
    
    try:
        logger.info("Server running on http://0.0.0.0:5000")

        while True:
            server.serve_forever(poll_interval=0.5)
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
    finally:
        logger.info("Server stopped")