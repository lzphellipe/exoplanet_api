
import logging
from flask import jsonify, request

from services.exoplanet_service import ExoplanetService
from services.csv_manager import CSVManager
from models.ai_memory import AIMemory

logger = logging.getLogger(__name__)

# Instâncias únicas dos serviços
exoplanet_service = ExoplanetService()
csv_manager = CSVManager()
ai_memory = AIMemory()


def configure_exoplanet_endpoints(app):
    """Configura todos os endpoints de exoplanetas"""

    @app.route('/api/exoplanets', methods=['GET'])
    def get_exoplanets():

        try:
            logger.info("Buscando dados de exoplanetas TESS...")

            data = exoplanet_service.get_tess_exoplanets()

            if data is None:
                return jsonify({
                    "error": "Falha ao buscar dados da API"
                }), 500

            # Armazena no CSV
            save_result = csv_manager.save_data(
                data=data,
                file_key='exoplanets',
                append=True,
                source='api'
            )

            if not save_result.get('success'):
                logger.warning(f"Erro ao salvar CSV: {save_result.get('error')}")

            return jsonify({
                "message": "Dados recuperados e armazenados com sucesso",
                "count": len(data),
                "csv_info": save_result,
                "memory_stats": ai_memory.get_memory_stats()
            })

        except Exception as e:
            logger.error(f"Erro no endpoint /api/exoplanets: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/exoplanets/confirmed', methods=['GET'])
    def get_confirmed_exoplanets():

        try:
            logger.info("Searching for confirmed exoplanets...")

            data = exoplanet_service.get_exoplanets_confirmed()

            if data is None:
                return jsonify({
                    "error": "Failed to fetch confirmed data"
                }), 404

            # Salva no CSV
            save_result = csv_manager.save_data(
                data=data,
                file_key='confirmed_planets',
                append=True,
                source='api'
            )

            return jsonify({
                "message": "Confirmed exoplanets successfully retrieved",
                "csv_info": save_result,
            })

        except Exception as e:
            logger.error(f"Erro no endpoint /api/exoplanets/confirmed: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/exoplanets/koi', methods=['GET'])
    def get_koi_candidates():

        try:

            data = exoplanet_service.fetch_koi_candidates()

            if data is None or len(data) == 0:
                return jsonify({
                    "error": "No KOI candidates found"
                }), 404

            # Salva no CSV (adicionar método save_koi_candidates ao csv_manager)
            try:
                save_result = csv_manager.save_koi_candidates(data)
                logger.info(f"✓ {len(data)} candidates saved in CSV")
            except Exception as e:
                logger.warning(f"Erro ao salvar KOI no CSV: {e}")
                save_result = {"success": False, "error": str(e)}

            return jsonify({
                "message": "KOI candidates successfully retrieved",
                "total": len(data),
                "csv_info": save_result,
                "data": data
            }), 200

        except Exception as e:
            logger.error(f"Error  endpoint /api/exoplanets/koi: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/exoplanets/<planet_name>', methods=['GET'])
    def get_exoplanet_by_name(planet_name):

        try:
            logger.info(f"Buscando exoplaneta: {planet_name}")

            planet = exoplanet_service.get_exoplanet_by_name(planet_name)

            if planet is None:
                return jsonify({
                    "error": f"Exoplaneta '{planet_name}' não encontrado"
                }), 404

            return jsonify({
                "message": "Exoplaneta encontrado",
                "data": planet
            })

        except Exception as e:
            logger.error(f"Erro ao buscar exoplaneta {planet_name}: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/memory/stats', methods=['GET'])
    def get_memory_stats():

        try:
            stats = ai_memory.get_memory_stats()

            return jsonify({
                "message": "Memory statistics retrieved",
                "stats": stats
            })

        except Exception as e:
            logger.error(f"Erro no endpoint /api/memory/stats: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/memory/clear', methods=['POST'])
    def clear_memory():

        try:
            ai_memory.clear_memory()

            return jsonify({
                "message": "AI sucesso"
            })

        except Exception as e:
            logger.error(f"Erro no endpoint /api/memory/clear: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/data/stats', methods=['GET'])
    def get_data_stats():

        try:
            stats = csv_manager.get_file_stats()
            summary = csv_manager.get_data_summary()

            return jsonify({
                "message": " sucess",
                "file_stats": stats,
                "data_summary": summary
            })

        except Exception as e:
            logger.error(f"Erro no endpoint /api/data/stats: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    logger.info("✓ Endpoints de exoplanetS CONFIGURED")