import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS

# Configuração do logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Criar aplicação Flask
app = Flask(__name__)

# Configuração do CORS
cors_origins = os.getenv('CORS_ORIGINS', '').split(',')
default_origins = [
    "https://exoplanet-lazaro.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

all_origins = list(set(default_origins + [o.strip() for o in cors_origins if o.strip()]))

CORS(app, resources={
    r"/*": {
        "origins": all_origins,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "supports_credentials": False,
        "max_age": 3600
    }
})

# Importar e configurar os endpoints dos controllers
try:
    from controllers.exoplanet_endpoints import configure_exoplanet_endpoints

    configure_exoplanet_endpoints(app)
    logger.info("✅ Exoplanet endpoints configurados")
except ImportError as e:
    logger.warning(f"⚠️ Não foi possível importar exoplanet_endpoints: {e}")

try:
    from controllers.lightkurve_endpoints import configure_lightkurve_endpoints

    configure_lightkurve_endpoints(app)
    logger.info("✅ Lightkurve endpoints configurados")
except ImportError as e:
    logger.warning(f"⚠️ Não foi possível importar lightkurve_endpoints: {e}")

try:
    from controllers.model_endpoints import configure_model_endpoints

    configure_model_endpoints(app)
    logger.info("✅ Model endpoints configurados")
except ImportError as e:
    logger.warning(f"⚠️ Não foi possível importar model_endpoints: {e}")


# Rota raiz
@app.route('/', methods=['GET'])
def home():
    """Rota principal da API"""
    return jsonify({
        "message": "Exoplanet API - Machine Learning Platform",
        "version": "2.0",
        "status": "running",
        "endpoints": {
            "exoplanets": {
                "GET /api/exoplanets": "Buscar dados de exoplanetas TESS",
                "GET /api/exoplanets/<planet_name>": "Buscar exoplaneta específico",
                "GET /api/exoplanets/confirmed": "Buscar exoplanetas confirmados",
                "GET /api/exoplanets/koi/<limit>": "Buscar candidatos KOI"
            },
            "lightcurves": {
                "GET /api/lightcurves/search/<target_name>": "Buscar light curves disponíveis",
                "GET /api/lightcurves/download/<target_name>": "Download de light curve",
                "GET /api/lightkurve/confirmed": "Processar features de light curves"
            },
            "model": {
                "POST /api/model/train": "Treinar modelo LightGBM",
                "POST /api/model/predict": "Fazer predições",
                "GET /api/model/info": "Informações do modelo",
                "GET /api/model/features/importance": "Importância das features",
                "GET /api/model/history": "Histórico de modelos",
                "POST /api/model/evaluate": "Avaliar modelo"
            },
            "memory": {
                "GET /api/memory/stats": "Estatísticas da memória da IA",
                "POST /api/memory/clear": "Limpar memória da IA"
            },
            "health": {
                "GET /health": "Health check da API"
            }
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Exoplanet ML API",
        "version": "2.0"
    }), 200


@app.route('/api/info', methods=['GET'])
def api_info():
    """Informações sobre a API"""
    return jsonify({
        "name": "Exoplanet Machine Learning API",
        "description": "API para análise de exoplanetas usando Machine Learning",
        "features": [
            "Busca de dados de exoplanetas da NASA Exoplanet Archive",
            "Análise de light curves usando Lightkurve",
            "Treinamento de modelos ML com LightGBM",
            "Predições e classificação de exoplanetas",
            "Sistema de memória para IA"
        ],
        "tech_stack": {
            "framework": "Flask",
            "ml": "LightGBM, scikit-learn",
            "data": "Pandas, NumPy",
            "astronomy": "Lightkurve"
        }
    })


# Manipuladores de erro
@app.errorhandler(404)
def not_found(error):
    """Manipulador de erro 404"""
    return jsonify({
        "error": "Endpoint não encontrado",
        "status": 404,
        "message": "O endpoint solicitado não existe. Veja / para lista de endpoints disponíveis."
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Manipulador de erro 500"""
    logger.error(f"Erro interno: {error}")
    return jsonify({
        "error": "Erro interno do servidor",
        "status": 500,
        "message": "Ocorreu um erro ao processar sua requisição."
    }), 500


@app.errorhandler(Exception)
def handle_exception(error):
    """Manipulador genérico de exceções"""
    logger.error(f"Exceção não tratada: {error}", exc_info=True)
    return jsonify({
        "error": "Erro inesperado",
        "status": 500,
        "message": str(error)
    }), 500


# Middleware para logging de requisições
@app.before_request
def log_request():
    """Log de cada requisição"""
    from flask import request
    logger.info(f"{request.method} {request.path}")


@app.after_request
def log_response(response):
    """Log de cada resposta"""
    from flask import request
    logger.info(f"{request.method} {request.path} - {response.status_code}")
    return response


if __name__ == '__main__':
    # Porta do ambiente ou padrão 5000
    port = int(os.environ.get('PORT', 5000))

    # Debug mode apenas em desenvolvimento
    debug_mode = os.getenv('FLASK_ENV') != 'production'

    logger.info(f"🚀 Iniciando servidor na porta {port}")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    logger.info(f"🌍 CORS origins: {all_origins}")

    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=port
    )