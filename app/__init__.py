from flask import Flask
from config import Config
import logging

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Logging ayarları
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('api.log', mode='w'),
            logging.StreamHandler()
        ]
    )
    app.logger.setLevel(logging.INFO)

    # Blueprint kaydı
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app