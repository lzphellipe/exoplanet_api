from flask import Blueprint, jsonify, request
import os
import logging
from services.lightkurve_service import LightkurveService
from services.ml_service import MLService
import joblib

logger = logging.getLogger(__name__)

kepler_bp = Blueprint('kepler', __name__, url_prefix='/api/kepler')


@kepler_bp.route('/fetch', methods=['GET'])
def fetch_kepler():
    """Fetch and process light curves for Kepler training dataset (koi_candidates.csv).

    Query Parameters:
        max_samples (int, optional): Maximum number of samples to process.
        mission (str): Data mission (default: 'Kepler').

    Returns:
        JSON: Processed light curve data with sample and file path.
    """
    max_samples = request.args.get('max_samples', type=int)
    mission = request.args.get('mission', 'Kepler')
    dataset_path = 'data/koi_candidates.csv'

    logger.info(f"Processing Kepler dataset: {dataset_path} (mission: {mission})")
    try:
        if not os.path.exists(dataset_path):
            return jsonify({
                "error": f"Dataset file {dataset_path} not found",
                "status": 400
            }), 400
        lightkurve_service = LightkurveService(data_dir="data")
        features_df = lightkurve_service.preprocess_train_data(dataset_path, max_samples, mission)
        return jsonify({
            "message": f"Processed light curves for {dataset_path}",
            "total_records": len(features_df),
            "sample": features_df.head(5).to_dict(orient="records"),
            "features_path": lightkurve_service.features_file
        }), 200
    except Exception as e:
        logger.error(f"Error processing Kepler dataset: {e}")
        return jsonify({
            "error": "Internal server error",
            "status": 500,
            "message": str(e)
        }), 500


@kepler_bp.route('/train', methods=['POST'])
def train_kepler():
    """Train a LightGBM model on Kepler training dataset (koi_candidates.csv).

    JSON Body:
        max_samples (int, optional): Maximum number of samples to process.
        mission (str): Data mission (default: 'Kepler').

    Returns:
        JSON: Training metrics and model file path.
    """
    data = request.get_json(silent=True) or {}  # Use silent=True to avoid 415 error
    max_samples = data.get('max_samples')
    mission = data.get('mission', 'Kepler')
    dataset_path = 'data/koi_candidates.csv'
    test_dataset_path = 'data/koi_cumulative.csv'
    model_path = 'models/koi_candidates_model.pkl'

    logger.info(f"Starting training for dataset: {dataset_path}")
    try:
        if not os.path.exists(dataset_path) or not os.path.exists(test_dataset_path):
            return jsonify({
                "error": f"Dataset file {dataset_path} or {test_dataset_path} not found",
                "status": 400
            }), 400
        lightkurve_service = LightkurveService(data_dir="data")
        ml_service = MLService(data_dir="data")

        # Preprocess training and test data
        data_train_final = lightkurve_service.preprocess_train_data(dataset_path, max_samples, mission)
        data_test_final = lightkurve_service.preprocess_test_data(test_dataset_path, max_samples, mission)

        # Train model
        result = ml_service.apply_lightgbm_model(data_train_final, data_test_final, model_path)
        return jsonify({
            "message": f"Model trained for {dataset_path}",
            "metrics": result["metrics"],
            "model_path": result["model_path"]
        }), 200
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return jsonify({
            "error": "Internal server error",
            "status": 500,
            "message": str(e)
        }), 500


@kepler_bp.route('/predict', methods=['POST'])
def predict_kepler():
    """Predict exoplanet classification for Kepler test dataset (koi_cumulative.csv).

    JSON Body:
        max_samples (int, optional): Maximum number of samples to process.
        mission (str): Data mission (default: 'Kepler').

    Returns:
        JSON: Predictions with identifiers, labels, and probabilities.
    """
    data = request.get_json(silent=True) or {}
    max_samples = data.get('max_samples')
    mission = data.get('mission', 'Kepler')
    dataset_path = 'data/koi_cumulative.csv'
    model_path = 'models/koi_candidates_model.pkl'

    logger.info(f"Making predictions for dataset: {dataset_path}")
    try:
        if not os.path.exists(dataset_path) or not os.path.exists(model_path):
            return jsonify({
                "error": f"Dataset {dataset_path} or model {model_path} not found",
                "status": 400
            }), 400
        lightkurve_service = LightkurveService(data_dir="data")
        ml_service = MLService(data_dir="data")

        # Preprocess test data
        data_test_final = lightkurve_service.preprocess_test_data(dataset_path, max_samples, mission)

        # Predict
        results_df = ml_service.predict(data_test_final, model_path)
        return jsonify({
            "message": f"Predictions made for {dataset_path}",
            "predictions": results_df.to_dict(orient="records")
        }), 200
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({
            "error": "Internal server error",
            "status": 500,
            "message": str(e)
        }), 500


@kepler_bp.route('/periodogram', methods=['POST'])
def periodogram_kepler():
    """Generate BLS periodograms for predicted exoplanet candidates.

    JSON Body:
        max_samples (int, optional): Maximum number of samples to process.
        mission (str): Data mission (default: 'Kepler').

    Returns:
        JSON: List of periodogram results with transit parameters and folded light curves.
    """
    data = request.get_json(silent=True) or {}
    max_samples = data.get('max_samples')
    mission = data.get('mission', 'Kepler')
    dataset_path = 'data/koi_cumulative.csv'
    model_path = 'models/koi_candidates_model.pkl'

    logger.info(f"Generating periodograms for dataset: {dataset_path}")
    try:
        if not os.path.exists(dataset_path) or not os.path.exists(model_path):
            return jsonify({
                "error": f"Dataset {dataset_path} or model {model_path} not found",
                "status": 400
            }), 400
        lightkurve_service = LightkurveService(data_dir="data")
        ml_service = MLService(data_dir="data")

        # Preprocess test data
        data_test_final = lightkurve_service.preprocess_test_data(dataset_path, max_samples, mission)

        # Generate predictions
        results_df = ml_service.predict(data_test_final, model_path)

        # Generate periodograms
        periodogram_results = lightkurve_service.periodogram_lightcurve(results_df, mission)

        return jsonify({
            "message": f"Periodograms generated for {len(periodogram_results)} candidates",
            "results": periodogram_results
        }), 200
    except Exception as e:
        logger.error(f"Error generating periodograms: {e}")
        return jsonify({
            "error": "Internal server error",
            "status": 500,
            "message": str(e)
        }), 500


def configure_exoplanet_endpoints(app):
    """Configure Kepler endpoints in the Flask application.

    Args:
        app (Flask): Flask application instance.
    """
    app.register_blueprint(kepler_bp)
    logger.info("Kepler endpoints configured successfully")