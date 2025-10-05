
from http.client import HTTPException
from flask import jsonify, request
import logging

from services.lightkurve_service import LightkurveService

logger = logging.getLogger(__name__)

lightkurve_service = LightkurveService()


def configure_lightkurve_endpoints(app):
    """Configura os endpoints de light curves"""

    @app.route('/api/lightcurves/search/<target_name>', methods=['GET'])
    def search_lightcurves(target_name):
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
                "message": "Light curves found",
                "data": result
            })

        except Exception as e:
            logger.error(f"Erro no endpoint /api/lightcurves/search/{target_name}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/lightcurves/download/<target_name>', methods=['GET'])
    def download_lightcurve(target_name):
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
                "message": "Light curve download sucessful",
                "data": result
            })

        except Exception as e:
            logger.error(f"Erro no endpoint /api/lightcurves/download/{target_name}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/lightkurve/confirmed", methods=['GET'])
    def fetch_lightkurve_features():
        try:
            features_df = lightkurve_service.process_dataset("confirmed", max_samples=10)

            return jsonify({
                "message": "Features de curvas de luz processadas",
                "features": features_df.to_dict(orient="records")
            })

        except Exception as e:
            logger.error(f"Erro ao processar curvas de luz: {e}")
            return jsonify({
                "error": str(e),
                "message": "Erro ao processar curvas de luz"
            }), 500

    logger.info("✓ Endpoints de lightkurve configurados")