import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from models.data_processor import DataProcessor
from models.ai_memory import AIMemory

logger = logging.getLogger(__name__)


class LightGBMService:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.ai_memory = AIMemory()
        self.model = None
        self.is_trained = False

    def train_model(self, training_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Treina o modelo LightGBM e salva na memória da IA"""
        try:
            # Busca dados para treinamento
            raw_data = self.ai_memory.get_training_data()
            if not raw_data:
                return {"error": "Nenhum dado de treinamento disponível na memória"}

            # Processa os dados
            processed_data = self.data_processor.process_data(raw_data)

            # Prepara dados para treinamento
            X, y, features = self.data_processor.prepare_training_data(processed_data)

            if X is None or len(X) == 0:
                return {"error": "Dados insuficientes para treinamento"}

            # Parâmetros padrão do modelo
            default_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'n_estimators': 100,
                'verbose': -1
            }

            # Atualiza com parâmetros fornecidos
            if training_params:
                default_params.update(training_params)

            # Treina o modelo
            self.model = lgb.LGBMClassifier(**default_params)
            self.model.fit(X, y)
            self.is_trained = True

            # Avaliação do modelo
            train_score = self.model.score(X, y)
            feature_importance = dict(zip(features, self.model.feature_importances_))

            # Salva na memória da IA
            model_info = {
                "model": self.model,
                "features": features,
                "training_params": default_params,
                "training_score": train_score,
                "feature_importance": feature_importance,
                "training_samples": len(X),
                "timestamp": pd.Timestamp.now().isoformat()
            }

            self.ai_memory.store_model(model_info)

            return {
                "message": "Modelo treinado e salvo na memória da IA com sucesso",
                "training_score": train_score,
                "feature_importance": feature_importance,
                "model_params": default_params,
                "samples_used": len(X),
                "features_count": len(features)
            }

        except Exception as e:
            logger.error(f"Erro no treinamento do modelo: {e}")
            return {"error": str(e)}

    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Faz predições usando o modelo treinado"""
        try:
            if not self.is_trained:
                # Tenta carregar da memória
                model_info = self.ai_memory.get_latest_model()
                if not model_info:
                    return {"error": "Nenhum modelo treinado disponível"}

                self.model = model_info["model"]
                self.is_trained = True

            expected_features = self.model.booster_.num_feature()
            if len(features) != expected_features:
                return {
                    "error": f"Número de features incorreto. Esperado: {expected_features}, Recebido: {len(features)}"
                }

            # Faz predição
            prediction = self.model.predict([features])
            probability = self.model.predict_proba([features])

            return {
                "prediction": int(prediction[0]),
                "probability": probability[0].tolist(),
                "class_probabilities": {
                    "class_0": float(probability[0][0]),
                    "class_1": float(probability[0][1])
                }
            }

        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {"error": str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo atual"""
        if not self.is_trained:
            return {"error": "Nenhum modelo treinado"}

        model_info = self.ai_memory.get_latest_model()
        if not model_info:
            return {"error": "Nenhuma informação do modelo na memória"}

        return {
            "is_trained": self.is_trained,
            "training_score": model_info.get("training_score"),
            "training_samples": model_info.get("training_samples"),
            "features_count": len(model_info.get("features", [])),
            "last_training": model_info.get("timestamp"),
            "model_parameters": model_info.get("training_params", {})
        }

    def get_feature_importance(self) -> Dict[str, Any]:
        """Retorna importância das features do modelo"""
        try:
            if not self.is_trained:
                return {"error": "Nenhum modelo treinado"}

            model_info = self.ai_memory.get_latest_model()
            feature_importance = model_info.get("feature_importance", {})

            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

            return {
                "feature_importance": sorted_importance,
                "top_features": list(sorted_importance.keys())[:5]
            }

        except Exception as e:
            logger.error(f"Erro ao obter importância das features: {e}")
            return {"error": str(e)}