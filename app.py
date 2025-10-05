import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importações locais - ajustar conforme sua estrutura
try:
    from models.data_processor import DataProcessor
    from utils.helpers import validate_parameters
except ImportError:
    logger.warning("Alguns módulos não puderam ser importados")

app = Flask(__name__)

# Configuração do CORS
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://exoplanet-lazaro.vercel.app",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "supports_credentials": False,
        "max_age": 3600
    }
})

# URLs das APIs
EXOPLANET_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars+where+disc_facility+like+%27%25TESS%25%27+order+by+pl_orbper+desc&format=json"
NASA_API_CONFIRMED = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,hostname,pl_orbper,pl_bmasse,pl_rade,st_teff,st_mass,st_rad,disc_year,disc_facility+from+ps+where+default_flag=1&format=json"
KOI_API = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+koi_period,koi_impact,koi_duration,koi_depth,koi_prad,koi_teq,koi_insol,koi_steff,koi_srad,koi_smass,koi_disposition+from+cumulative&format=json"


class ExoplanetAPI:
    def __init__(self):
        try:
            self.data_processor = DataProcessor()
        except:
            self.data_processor = None
        self.model = None
        self.features = None

    def fetch_exoplanet_data(self):
        """Busca dados de exoplanetas da API"""
        try:
            logger.info("Buscando dados de exoplanetas...")
            response = requests.get(EXOPLANET_URL, timeout=30)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Dados recuperados: {len(data)} registros")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao buscar dados: {e}")
            return None

    def fetch_exoplanet_koi(self):
        """Busca dados KOI de exoplanetas"""
        try:
            logger.info("Buscando dados KOI...")
            response = requests.get(KOI_API, timeout=30)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Dados KOI recuperados: {len(data)} registros")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao buscar dados KOI: {e}")
            return None

    def fetch_exoplanet_confirmed(self):
        """Busca dados de exoplanetas confirmados"""
        try:
            logger.info("Buscando dados de exoplanetas confirmados...")
            response = requests.get(NASA_API_CONFIRMED, timeout=30)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Dados confirmados recuperados: {len(data)} registros")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao buscar dados confirmados: {e}")
            return None


# Instância da API
exoplanet_api = ExoplanetAPI()


@app.route('/', methods=['GET'])
def home():
    """Rota raiz"""
    return jsonify({
        "message": "Exoplanet API",
        "version": "1.0",
        "endpoints": {
            "exoplanets": "/api/exoplanets",
            "confirmed": "/api/exoplanets/confirmed",
            "koi": "/api/exoplanets/koi",
            "health": "/health"
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check da API"""
    return jsonify({
        "status": "healthy",
        "service": "Exoplanet API",
        "version": "1.0"
    })


@app.route('/api/exoplanets', methods=['GET'])
def get_exoplanets():
    """Endpoint para buscar dados brutos de exoplanetas"""
    try:
        data = exoplanet_api.fetch_exoplanet_data()
        if data is None:
            return jsonify({"error": "Falha ao buscar dados"}), 500

        return jsonify({
            "message": "Dados recuperados com sucesso",
            "count": len(data),
            "data": data
        })
    except Exception as e:
        logger.error(f"Erro no endpoint /api/exoplanets: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/exoplanets/koi', methods=['GET'])
def get_exoplanets_koi():
    """Endpoint para buscar dados KOI"""
    try:
        data = exoplanet_api.fetch_exoplanet_koi()
        if data is None:
            return jsonify({"error": "Falha ao buscar dados KOI"}), 500

        return jsonify({
            "message": "Dados KOI recuperados com sucesso",
            "count": len(data),
            "data": data
        })
    except Exception as e:
        logger.error(f"Erro no endpoint /api/exoplanets/koi: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/exoplanets/confirmed', methods=['GET'])
def fetch_confirmed():
    """Endpoint para buscar dados confirmados"""
    try:
        data = exoplanet_api.fetch_exoplanet_confirmed()
        if data is None:
            return jsonify({"error": "Falha ao buscar dados confirmados"}), 500

        return jsonify({
            "message": "Dados confirmados recuperados com sucesso",
            "count": len(data),
            "data": data
        })
    except Exception as e:
        logger.error(f"Erro no endpoint /api/exoplanets/confirmed: {e}")
        return jsonify({"error": str(e)}), 500


# Manipulador de erros global
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint não encontrado"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Erro interno do servidor"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)