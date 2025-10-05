from flask import jsonify, request
import logging
from services.lightGBM_service import LightGBMService
from models.ai_memory import AIMemory

logger = logging.getLogger(__name__)

model_service = LightGBMService()
ai_memory = AIMemory()


def configure_model_endpoints(app):
    """Configure ML model endpoints"""

    @app.route('/api/model/train', methods=['GET', 'POST'])
    def train_model():
        """Train LightGBM model - supports both GET and POST"""

        # If GET request, return training instructions
        if request.method == 'GET':
            return jsonify({
                "message": "To train the model, send a POST request to this endpoint",
                "method": "POST",
                "endpoint": "/api/model/train",
                "note": "You can send an empty body or custom parameters",
                "examples": {
                    "curl_empty": "curl -X POST http://localhost:5000/api/model/train",
                    "curl_with_json": "curl -X POST http://localhost:5000/api/model/train -H 'Content-Type: application/json' -d '{\"learning_rate\": 0.05}'",
                    "python": "requests.post('http://localhost:5000/api/model/train')"
                },
                "optional_parameters": {
                    "learning_rate": "float (default: 0.05)",
                    "n_estimators": "int (default: 100)",
                    "num_leaves": "int (default: 31)"
                },
                "current_status": model_service.get_model_info() if model_service.is_trained else {
                    "status": "No model trained yet"}
            }), 200

        # POST request - actual training
        try:
            logger.info("Starting model training...")

            # Safely get JSON data - handle empty body
            training_params = {}
            try:
                if request.data:  # Check if there's any data
                    training_params = request.get_json(silent=True) or {}
                logger.info(f"Training parameters received: {training_params}")
            except Exception as e:
                logger.warning(f"Could not parse JSON (using defaults): {e}")
                training_params = {}

            result = model_service.train_model(training_params)

            if 'error' in result:
                return jsonify(result), 400

            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error in endpoint /api/model/train: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/model/predict', methods=['POST'])
    def predict():
        """Make predictions with trained model"""
        try:
            # Check if request has data
            if not request.data:
                return jsonify({
                    "error": "Request body is required",
                    "example": {
                        "features": [1.0, 2.0, 3.0, "..."]
                    }
                }), 400

            data = request.get_json(silent=True)
            if not data or 'features' not in data:
                return jsonify({
                    "error": "Feature data required in request body",
                    "required_format": {
                        "features": [1.0, 2.0, 3.0, "..."]
                    },
                    "example_curl": "curl -X POST http://localhost:5000/api/model/predict -H 'Content-Type: application/json' -d '{\"features\": [1.0, 2.0, 3.0]}'"
                }), 400

            features = data['features']

            if not isinstance(features, list):
                return jsonify({"error": "Features must be a list"}), 400

            result = model_service.predict(features)

            if 'error' in result:
                return jsonify(result), 400

            # Store in AI memory
            prediction_record = {
                "features": features,
                "prediction": result["prediction"],
                "probability": result.get("probability"),
                "timestamp": result.get("timestamp")
            }
            ai_memory.store_prediction(prediction_record)

            return jsonify({
                "message": "Prediction completed successfully",
                "prediction": result,
                "memory_stats": ai_memory.get_memory_stats()
            }), 200

        except Exception as e:
            logger.error(f"Error in endpoint /api/model/predict: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/model/info', methods=['GET'])
    def get_model_info():
        """Return trained model information"""
        try:
            result = model_service.get_model_info()

            if 'error' in result:
                return jsonify(result), 404

            return jsonify({
                "message": "Model information",
                "data": result
            }), 200

        except Exception as e:
            logger.error(f"Error in endpoint /api/model/info: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/model/features/importance', methods=['GET'])
    def get_feature_importance():
        """Return feature importance"""
        try:
            result = model_service.get_feature_importance()

            if 'error' in result:
                return jsonify(result), 404

            return jsonify({
                "message": "Feature importance",
                "data": result
            }), 200

        except Exception as e:
            logger.error(f"Error in endpoint /api/model/features/importance: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/model/history', methods=['GET'])
    def get_model_history():
        """Return model training history"""
        try:
            history = ai_memory.get_model_history()
            return jsonify({
                "message": "Model history",
                "count": len(history),
                "data": history
            }), 200

        except Exception as e:
            logger.error(f"Error in endpoint /api/model/history: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/model/evaluate', methods=['POST'])
    def evaluate_model():
        """Evaluate trained model with test data"""
        try:
            # Handle empty body
            data = {}
            if request.data:
                data = request.get_json(silent=True) or {}

            test_data = data.get('test_data') if data else None

            result = model_service.evaluate_model(test_data)

            if 'error' in result:
                return jsonify(result), 400

            return jsonify({
                "message": "Model evaluation completed successfully",
                "evaluation": result
            }), 200

        except Exception as e:
            logger.error(f"Error in endpoint /api/model/evaluate: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    logger.info("✓ Model endpoints configured")