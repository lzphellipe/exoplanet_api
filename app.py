import json
import os
from http.client import HTTPException

from flask import Flask, request, jsonify
import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightkurve import search_lightcurve, config
import joblib
import logging
from models.data_processor import DataProcessor
from utils.helpers import validate_parameters

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# URL da API de exoplanetas
EXOPLANET_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars+where+disc_facility+like+%27%25TESS%25%27+order+by+pl_orbper+desc&format=json"
NASA_API_CONFIRMED = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,hostname,pl_orbper,pl_bmasse,pl_rade,st_teff,st_mass,st_rad,disc_year,disc_facility+from+ps+where+default_flag=1&format=json"

class ExoplanetAPI:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model = None
        self.features = None

    def fetch_exoplanet_data(app):
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

    def fetch_exoplanet_confirmed(app):
        """Busca dados de exoplanetas da API"""
        try:
            logger.info("Buscando dados de exoplanetas confirmados ...")
            response = requests.get(NASA_API_CONFIRMED, timeout=30)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Dados recuperados: {len(data)} registros")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao buscar dados: {e}")
            return None


# Instância da API
exoplanet_api = ExoplanetAPI()


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


@app.route('/api/exoplanets/processed', methods=['GET'])
def get_processed_exoplanets():
    """Endpoint para dados processados"""
    try:
        raw_data = exoplanet_api.fetch_exoplanet_data()
        if raw_data is None:
            return jsonify({"error": "Falha ao buscar dados"}), 500

        # Processa os dados
        processed_data = exoplanet_api.data_processor.process_data(raw_data)

        return jsonify({
            "message": "Dados processados com sucesso",
            "count": len(processed_data),
            "data": processed_data.to_dict(orient='records')
        })
    except Exception as e:
        logger.error(f"Erro no endpoint /api/exoplanets/processed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/lightcurves/<target_name>', methods=['GET'])
def get_lightcurve(target_name):
    """Busca light curves do LightKurve"""
    try:
        logger.info(f"Buscando light curve para: {target_name}")

        # Parâmetros da requisição
        mission = request.args.get('mission', 'TESS')
        quarter = request.args.get('quarter', None)
        campaign = request.args.get('campaign', None)

        # Busca light curves
        search_result = search_lightcurve(target_name, mission=mission)

        if len(search_result) == 0:
            return jsonify({"error": f"Nenhuma light curve encontrada para {target_name}"}), 404

        # Filtra por quarter/campaign se especificado
        if quarter:
            search_result = search_result[search_result.quarter == int(quarter)]
        if campaign:
            search_result = search_result[search_result.campaign == int(campaign)]

        if len(search_result) == 0:
            return jsonify({"error": "Nenhuma light curve encontrada com os filtros especificados"}), 404

        # Pega a primeira light curve
        lc = search_result[0].download()

        # Prepara dados para resposta
        lightcurve_data = {
            "target": target_name,
            "mission": mission,
            "time": lc.time.value.tolist(),
            "flux": lc.flux.value.tolist(),
            "flux_err": lc.flux_err.value.tolist() if lc.flux_err is not None else None,
            "cadence_no": lc.cadenceno.value.tolist() if lc.cadenceno is not None else None
        }

        return jsonify({
            "message": "Light curve recuperada com sucesso",
            "data": lightcurve_data
        })

    except Exception as e:
        logger.error(f"Erro ao buscar light curve: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/train/model', methods=['POST'])
def train_model():
    """Treina modelo LightGBM com os dados"""
    try:
        # Busca e processa dados
        raw_data = exoplanet_api.fetch_exoplanet_data()
        if raw_data is None:
            return jsonify({"error": "Falha ao buscar dados para treinamento"}), 500

        processed_data = exoplanet_api.data_processor.process_data(raw_data)

        # Prepara dados para treinamento
        X, y, features = exoplanet_api.data_processor.prepare_training_data(processed_data)

        if X is None or len(X) == 0:
            return jsonify({"error": "Dados insuficientes para treinamento"}), 400

        # Parâmetros do modelo da requisição
        params = request.get_json() or {}
        model_params = {
            'objective': params.get('objective', 'binary'),
            'metric': params.get('metric', 'binary_logloss'),
            'boosting_type': params.get('boosting_type', 'gbdt'),
            'num_leaves': params.get('num_leaves', 31),
            'learning_rate': params.get('learning_rate', 0.05),
            'feature_fraction': params.get('feature_fraction', 0.9),
            'n_estimators': params.get('n_estimators', 100),
            'verbose': -1
        }

        # Treina o modelo
        model = lgb.LGBMClassifier(**model_params)
        model.fit(X, y)

        # Salva o modelo
        model_filename = "exoplanet_model.pkl"
        joblib.dump(model, model_filename)
        exoplanet_api.model = model
        exoplanet_api.features = features

        # Avaliação do modelo
        train_score = model.score(X, y)
        feature_importance = dict(zip(features, model.feature_importances_))

        return jsonify({
            "message": "Modelo treinado com sucesso",
            "training_score": train_score,
            "feature_importance": feature_importance,
            "model_params": model_params,
            "samples_used": len(X)
        })

    except Exception as e:
        logger.error(f"Erro no treinamento do modelo: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Faz previsões com o modelo treinado"""
    try:
        if exoplanet_api.model is None:
            return jsonify({"error": "Modelo não treinado. Execute /api/train/model primeiro"}), 400

        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Dados de features necessários"}), 400

        # Prepara features para predição
        input_features = data['features']
        if len(input_features) != len(exoplanet_api.features):
            return jsonify({
                "error": f"Número de features incorreto. Esperado: {len(exoplanet_api.features)}, Recebido: {len(input_features)}",
                "expected_features": exoplanet_api.features
            }), 400

        # Faz predição
        prediction = exoplanet_api.model.predict([input_features])
        probability = exoplanet_api.model.predict_proba([input_features])

        return jsonify({
            "prediction": int(prediction[0]),
            "probability": probability[0].tolist(),
            "features_used": exoplanet_api.features
        })

    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/features/importance', methods=['GET'])
def get_feature_importance():
    """Retorna importância das features do modelo"""
    try:
        if exoplanet_api.model is None:
            return jsonify({"error": "Modelo não treinado"}), 400

        importance_dict = dict(zip(exoplanet_api.features, exoplanet_api.model.feature_importances_))
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return jsonify({
            "feature_importance": sorted_importance
        })
    except Exception as e:
        logger.error(f"Erro ao obter importância das features: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check da API"""
    return jsonify({"status": "healthy", "service": "Exoplanet API"})

@app.route('/api/exoplanets/confirmed', methods=['GET'])
def fetch_confirmed():
    try:
        data = exoplanet_api.fetch_exoplanet_confirmed()
        if data is None:
            return jsonify({"error": "Falha ao buscar dados"}), 500

        return jsonify({
            "message": "Dados recuperados com sucesso",
            "count": len(data),
            "data": data
        })
    except Exception as e:
        logger.error(f"Erro no endpoint /exoplanets/confirmed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)