import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from controllers import kepler_endpoints

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__)

# Configure CORS
cors_origins = os.getenv('CORS_ORIGINS', '').split(',')
default_origins = [
    "https://exoplanet-lazaro.vercel.app",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
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
    logger.info("✅ ")
except ImportError as e:
    logger.warning(f"⚠️  exoplanet_endpoints: {e}")

try:
    from controllers.lightkurve_endpoints import configure_lightkurve_endpoints

    configure_lightkurve_endpoints(app)
    logger.info("✅ Lightkurve ")
except ImportError as e:
    logger.warning(f"⚠️  lightkurve_endpoints: {e}")

try:
    from controllers.model_endpoints import configure_model_endpoints

    configure_model_endpoints(app)
    logger.info("✅ Model ")
except ImportError as e:
    logger.warning(f"⚠️  {e}")


# Configure Kepler endpoints
try:
    kepler_endpoints.configure_exoplanet_endpoints(app)
    logger.info("✅ Kepler endpoints configured successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import kepler_endpoints: {e}")

# Root endpoint
@app.route('/', methods=['GET'])
def home():
    """Root endpoint providing API information."""
    return jsonify({
        "message": "Exoplanet API - Machine Learning Platform",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "kepler": {
                "GET /api/kepler/fetch": "Fetch and process Kepler light curves",
                "POST /api/kepler/train": "Train LightGBM model on Kepler data",
                "POST /api/kepler/predict": "Predict exoplanet classifications"
            },
            "health": {
                "GET /health": "Health check of the API"
            }
        }
    })

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Exoplanet ML API",
        "version": "1.0"
    }), 200

# API info endpoint
@app.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint."""
    return jsonify({
        "name": "Exoplanet Machine Learning API",
        "description": "API for exoplanet analysis using machine learning",
        "features": [
            "Processing Kepler light curves",
            "Training LightGBM models",
            "Predicting exoplanet classifications"
        ],
        "tech_stack": {
            "framework": "Flask",
            "ml": "LightGBM, scikit-learn",
            "data": "Pandas, NumPy",
            "astronomy": "Lightkurve"
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "status": 404,
        "message": "The requested endpoint does not exist. See / for available endpoints."
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "status": 500,
        "message": "An error occurred while processing your request."
    }), 500

@app.errorhandler(Exception)
def handle_exception(error):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {error}", exc_info=True)
    return jsonify({
        "error": "Unexpected error",
        "status": 500,
        "message": str(error)
    }), 500

# Request logging middleware
@app.before_request
def log_request():
    """Log each incoming request."""
    from flask import request
    logger.info(f"{request.method} {request.path}")

@app.after_request
def log_response(response):
    """Log each response."""
    from flask import request
    logger.info(f"{request.method} {request.path} - {response.status_code}")
    return response

if __name__ == '__main__':
    # Port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Debug mode only in development
    debug_mode = os.getenv('FLASK_ENV') != 'production'

    logger.info(f"🚀 Starting server on port {port}")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    logger.info(f"🌍 CORS origins: {all_origins}")

    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=port
    )