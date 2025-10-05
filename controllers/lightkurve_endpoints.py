from flask import jsonify, request
import logging
from services.lightkurve_service import LightkurveService

logger = logging.getLogger(__name__)

lightkurve_service = LightkurveService()


def configure_lightkurve_endpoints(app):
    """Configura os endpoints de light curves"""

    @app.route('/api/lightcurves/search/<target_name>', methods=['GET'])
    def search_lightcurves(target_name):
        """Busca light curves disponíveis para um alvo"""
        try:
            mission = request.args.get('mission', 'TESS')
            quarter = request.args.get('quarter', None, type=int)
            campaign = request.args.get('campaign', None, type=int)

            result = lightkurve_service.search_lightcurves(
                target_name, mission, quarter, campaign
            )

            if 'error' in result:
                return jsonify(result), 404

            return jsonify({
                "message": "Light curves encontradas",
                "data": result
            })
        except Exception as e:
            logger.error(f"Erro no endpoint /api/lightcurves/search/{target_name}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/lightcurves/download/<target_name>', methods=['GET'])
    def download_lightcurve(target_name):
        """Faz download de uma light curve específica"""
        try:
            mission = request.args.get('mission', 'TESS')
            quarter = request.args.get('quarter', None, type=int)
            campaign = request.args.get('campaign', None, type=int)
            index = request.args.get('index', 0, type=int)

            result = lightkurve_service.download_lightcurve(
                target_name, mission, index, quarter, campaign
            )

            if 'error' in result:
                return jsonify(result), 404

            return jsonify({
                "message": "Light curve baixada com sucesso",
                "data": result
            })
        except Exception as e:
            logger.error(f"Erro no endpoint /api/lightcurves/download/{target_name}: {e}")
            return jsonify({"error": str(e)}), 500