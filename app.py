"""
Aplicação Flask Principal - Exoplanet API
Refatorado: Imports circulares corrigidos
"""
import logging
from flask import Flask, jsonify
from flask_cors import CORS

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializa aplicação Flask
app = Flask(__name__)

# Configuração CORS
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://exoplanet-lazaro.vercel.app",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
            "http://127.0.0.1:5000"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "supports_credentials": False,
        "max_age": 3600
    }
})

# ⚠️ IMPORTANTE: Importar APÓS criar a instância 'app'
# Isso evita import circular
try:
    from controllers.exoplanet_endpoints import configure_exoplanet_endpoints
    from controllers.lightkurve_endpoints import configure_lightkurve_endpoints
    from controllers.model_endpoints import configure_model_endpoints

    # Configura todos os endpoints
    configure_exoplanet_endpoints(app)
    configure_lightkurve_endpoints(app)
    configure_model_endpoints(app)

    logger.info("✓ Todos os endpoints configurados com sucesso")

except ImportError as e:
    logger.error(f"Erro ao importar endpoints: {e}")
    logger.error("Verifique se os arquivos estão na pasta 'controllers/'")
    raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check da API"""
    try:
        from services.csv_manager import CSVManager

        csv_manager = CSVManager()
        stats = csv_manager.get_file_stats()

        return jsonify({
            "status": "healthy",
            "service": "Exoplanet API",
            "version": "2.0.0",
            "endpoints": {
                "exoplanets": "✓",
                "lightkurve": "✓",
                "model": "✓"
            },
            "data_files": {
                "exoplanets": stats['files'].get('exoplanets', {}).get('exists', False),
                "confirmed_planets": stats['files'].get('confirmed_planets', {}).get('exists', False)
            }
        })
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        return jsonify({
            "status": "degraded",
            "error": str(e)
        }), 500


@app.route('/', methods=['GET'])
def root():
    """Rota raiz com informações da API"""
    return jsonify({
        "message": "Exoplanet API v2.0.0",
        "status": "running",
        "documentation": "/health",
        "available_endpoints": {
            "exoplanets": [
                "GET /api/exoplanets",
                "GET /api/exoplanets/confirmed",
                "GET /api/exoplanets/koi/<limit>",
                "GET /api/exoplanets/<planet_name>"
            ],
            "memory": [
                "GET /api/memory/stats",
                "POST /api/memory/clear"
            ],
            "data": [
                "GET /api/data/stats"
            ],
            "model": [
                "POST /api/model/train",
                "POST /api/model/predict",
                "GET /api/model/info",
                "GET /api/model/features/importance",
                "GET /api/model/history",
                "POST /api/model/evaluate"
            ],
            "lightcurves": [
                "GET /api/lightcurves/search/<target_name>",
                "GET /api/lightcurves/download/<target_name>",
                "GET /api/lightkurve/confirmed"
            ]
        }
    })


@app.errorhandler(404)
def not_found(error):
    """Handler para rotas não encontradas"""
    return jsonify({
        "error": "Endpoint não encontrado",
        "message": str(error),
        "tip": "Acesse GET / para ver todos os endpoints disponíveis"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handler para erros internos"""
    logger.error(f"Erro interno: {error}")
    return jsonify({
        "error": "Erro interno do servidor",
        "message": "Por favor, verifique os logs ou contate o administrador"
    }), 500


@app.errorhandler(Exception)
def handle_exception(error):
    """Handler genérico para exceções não tratadas"""
    logger.error(f"Exceção não tratada: {error}", exc_info=True)
    return jsonify({
        "error": "Erro inesperado",
        "message": str(error),
        "type": type(error).__name__
    }), 500


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Iniciando Exoplanet API v2.0.0")
    logger.info("=" * 60)
    logger.info("📊 Endpoints configurados:")
    logger.info("   ✓ Exoplanets endpoints")
    logger.info("   ✓ Lightkurve endpoints")
    logger.info("   ✓ Model endpoints")
    logger.info("=" * 60)
    logger.info("🌐 Servidor rodando em: http://localhost:5000")
    logger.info("📖 Documentação: http://localhost:5000/")
    logger.info("💚 Health check: http://localhost:5000/health")
    logger.info("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)