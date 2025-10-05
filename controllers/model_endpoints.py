from flask import jsonify, request
import logging
from services.model_service import ModelService
from models.ai_memory import AIMemory

logger = logging.getLogger(__name__)

model_service = ModelService()
ai_memory = AIMemory()


def configure_model_endpoints(app):
    """Configura os endpoints de modelo ML"""

    @app.route('/api/model/train', methods=['POST'])
    def train_model():
        """Treina o modelo LightGBM"""
        try:
            training_params = request.get_json() or {}

            result = model_service.train_model(training_params)

            if 'error' in result:
                return jsonify(result), 400

            return jsonify(result)
        except Exception as e:
            logger.error(f"Erro no endpoint /api/model/train: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/model/predict', methods=['POST'])
    def predict():
        """Faz predições com o modelo treinado"""
        try:
            data = request.get_json()
            if not data or 'features' not in data:
                return jsonify({"error": "Dados de features necessários"}), 400

            features = data['features']

            if not isinstance(features, list):
                return jsonify({"error": "Features devem ser uma lista"}), 400

            result = model_service.predict(features)

            if 'error' in result:
                return jsonify(result), 400

            # Armazena na memória da IA
            prediction_record = {
                "features": features,
                "prediction": result["prediction"],
                "probability": result["probability"],
                "timestamp": result.get("timestamp")
            }
            ai_memory.store_prediction(prediction_record)

            return jsonify({
                "message": "Predição realizada com sucesso",
                "prediction": result,
                "memory_stats": ai_memory.get_memory_stats()
            })
        except Exception as e:
            logger.error(f"Erro no endpoint /api/model/predict: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/model/info', methods=['GET'])
    def get_model_info():
        """Retorna informações do modelo treinado"""
        try:
            result = model_service.get_model_info()

            if 'error' in result:
                return jsonify(result), 404

            return jsonify({
                "message": "Informações do modelo",
                "data": result
            })
        except Exception as e:
            logger.error(f"Erro no endpoint /api/model/info: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/model/features/importance', methods=['GET'])
    def get_feature_importance():
        """Retorna importância das features"""
        try:
            result = model_service.get_feature_importance()

            if 'error' in result:
                return jsonify(result), 404

            return jsonify({
                "message": "Importância das features",
                "data": result
            })
        except Exception as e:
            logger.error(f"Erro no endpoint /api/model/features/importance: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/model/history', methods=['GET'])
    def get_model_history():
        """Retorna histórico de modelos treinados"""
        try:
            history = ai_memory.get_model_history()
            return jsonify({
                "message": "Histórico de modelos",
                "count": len(history),
                "data": history
            })
        except Exception as e:
            logger.error(f"Erro no endpoint /api/model/history: {e}")
            return jsonify({"error": str(e)}), 500


