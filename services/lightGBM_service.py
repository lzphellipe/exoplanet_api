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
        self.problem_type = None

    def train_model(self, training_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Treina o modelo LightGBM (classificação ou regressão) com fallbacks"""
        try:
            logger.info("Iniciando treinamento do modelo...")

            # Busca dados para treinamento
            from services.exoplanet_service import ExoplanetService
            exoplanet_service = ExoplanetService()

            raw_data = self.ai_memory.get_training_data()

            # Se não há dados na memória, busca da API
            if not raw_data:
                logger.info("Nenhum dado na memória, buscando da API...")
                raw_data = exoplanet_service.get_tess_exoplanets()

                # Se a API falhar, usa dados de exemplo
                if not raw_data:
                    logger.warning("API falhou, usando dados de exemplo para desenvolvimento")
                    raw_data = exoplanet_service.get_sample_data()
                    self.ai_memory.store_training_data(raw_data)

            if not raw_data:
                error_msg = "Nenhum dado de treinamento disponível"
                logger.error(error_msg)
                return {"error": error_msg}

            logger.info(f"Dados brutos carregados: {len(raw_data)} registros")

            # Processa os dados
            processed_data = self.data_processor.process_data(raw_data)

            if processed_data is None or processed_data.empty:
                error_msg = "Falha ao processar dados"
                logger.error(error_msg)
                return {"error": error_msg}

            logger.info(f"Dados processados: {processed_data.shape}")

            # Prepara dados para treinamento
            X, y, features = self.data_processor.prepare_training_data(processed_data)

            if X is None or len(X) == 0:
                error_msg = "Dados insuficientes para treinamento após processamento"
                logger.error(error_msg)
                return {"error": error_msg}

            logger.info(f"Dados preparados para treinamento. X: {X.shape}, y: {y.shape}")

            # Obtém informações do problema
            problem_info = self.data_processor.get_problem_info()
            self.problem_type = problem_info['problem_type']

            logger.info(f"Treinando modelo para: {self.problem_type}")
            logger.info(f"Target: {problem_info['target_column']}")

            # Parâmetros padrão baseados no tipo de problema
            if self.problem_type == 'classification':
                n_classes = len(np.unique(y))
                default_params = {
                    'objective': 'multiclass' if n_classes > 2 else 'binary',
                    'metric': 'multi_logloss' if n_classes > 2 else 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': min(31, len(X)),  # Ajusta baseado no tamanho dos dados
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'n_estimators': min(100, len(X) // 2),  # Ajusta baseado no tamanho dos dados
                    'verbose': -1,
                }

                if n_classes > 2:
                    default_params['num_class'] = n_classes

                logger.info(f"Classificação com {n_classes} classes")
            else:  # regression
                default_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': min(31, len(X)),
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'n_estimators': min(100, len(X) // 2),
                    'verbose': -1
                }
                logger.info("Regressão configurada")

            # Atualiza com parâmetros fornecidos
            if training_params:
                default_params.update(training_params)
                logger.info(f"Parâmetros personalizados: {training_params}")

            logger.info(f"Parâmetros finais do modelo: {default_params}")

            # Treina o modelo apropriado
            if self.problem_type == 'classification':
                self.model = lgb.LGBMClassifier(**default_params)
            else:
                self.model = lgb.LGBMRegressor(**default_params)

            logger.info("Iniciando fit do modelo...")
            self.model.fit(X, y)
            self.is_trained = True
            logger.info("Modelo treinado com sucesso!")

            # Avaliação do modelo
            if self.problem_type == 'classification':
                train_score = self.model.score(X, y)
                y_pred = self.model.predict(X)
                accuracy = np.mean(y == y_pred)
                logger.info(f"Acurácia do treinamento: {accuracy:.4f}")
            else:
                from sklearn.metrics import r2_score, mean_squared_error
                y_pred = self.model.predict(X)
                train_score = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                logger.info(f"R² do treinamento: {train_score:.4f}, RMSE: {rmse:.4f}")

            feature_importance = dict(zip(features, self.model.feature_importances_))

            # Salva na memória da IA
            model_info = {
                "model": self.model,
                "features": features,
                "training_params": default_params,
                "training_score": train_score,
                "feature_importance": feature_importance,
                "training_samples": len(X),
                "problem_type": self.problem_type,
                "target_column": problem_info['target_column'],
                "timestamp": pd.Timestamp.now().isoformat()
            }

            # Adiciona métricas específicas
            if self.problem_type == 'classification':
                model_info["accuracy"] = accuracy
                model_info["class_distribution"] = np.bincount(y).tolist()
                model_info["n_classes"] = n_classes
            else:
                model_info["rmse"] = rmse
                model_info["r2_score"] = train_score

            self.ai_memory.store_model(model_info)

            response = {
                "message": f"Modelo de {self.problem_type} treinado e salvo com sucesso",
                "problem_type": self.problem_type,
                "target_column": problem_info['target_column'],
                "training_score": train_score,
                "feature_importance": feature_importance,
                "model_params": default_params,
                "samples_used": len(X),
                "features_count": len(features),
                "data_source": "API" if raw_data != exoplanet_service.get_sample_data() else "Sample Data"
            }

            # Adiciona métricas específicas
            if self.problem_type == 'classification':
                response["accuracy"] = accuracy
                response["classes"] = n_classes
                response["class_distribution"] = np.bincount(y).tolist()
            else:
                response["r2_score"] = train_score
                response["rmse"] = rmse

            logger.info("Treinamento concluído com sucesso!")
            return response

        except Exception as e:
            logger.error(f"Erro no treinamento do modelo: {e}", exc_info=True)
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
                self.problem_type = model_info.get("problem_type", "regression")

            expected_features = len(self.model.feature_name_)
            if len(features) != expected_features:
                return {
                    "error": f"Número de features incorreto. Esperado: {expected_features}, Recebido: {len(features)}",
                    "expected_features": self.model.feature_name_
                }

            # Faz predição
            prediction = self.model.predict([features])
            probability = None

            if self.problem_type == 'classification':
                probability = self.model.predict_proba([features])
                # Converte para int se for classificação
                prediction = int(prediction[0])
                class_probabilities = {
                    f"class_{i}": float(prob)
                    for i, prob in enumerate(probability[0])
                }
            else:
                prediction = float(prediction[0])
                class_probabilities = {}

            return {
                "prediction": prediction,
                "probability": probability[0].tolist() if probability is not None else None,
                "class_probabilities": class_probabilities,
                "problem_type": self.problem_type,
                "confidence": float(max(probability[0])) if probability is not None else None
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

        info = {
            "is_trained": self.is_trained,
            "problem_type": model_info.get("problem_type"),
            "training_score": model_info.get("training_score"),
            "training_samples": model_info.get("training_samples"),
            "features_count": len(model_info.get("features", [])),
            "last_training": model_info.get("timestamp"),
            "model_parameters": model_info.get("training_params", {})
        }

        # Adiciona métricas específicas
        if model_info.get("problem_type") == 'classification':
            info["accuracy"] = model_info.get("accuracy")
            info["classes"] = len(model_info.get("class_distribution", []))
        else:
            info["r2_score"] = model_info.get("r2_score")
            info["rmse"] = model_info.get("rmse")

        return info

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
                "top_features": list(sorted_importance.keys())[:5],
                "problem_type": model_info.get("problem_type")
            }

        except Exception as e:
            logger.error(f"Erro ao obter importância das features: {e}")
            return {"error": str(e)}

    def evaluate_model(self, test_data: List[Dict] = None) -> Dict[str, Any]:
        """Avalia o modelo com dados de teste"""
        try:
            if not self.is_trained:
                return {"error": "Nenhum modelo treinado"}

            # Se não fornecer dados de teste, usa dados de treinamento da memória
            if test_data is None:
                raw_data = self.ai_memory.get_training_data()
                if not raw_data:
                    return {"error": "Nenhum dado disponível para avaliação"}
            else:
                raw_data = test_data

            # Processa os dados
            processed_data = self.data_processor.process_data(raw_data)
            X, y, features = self.data_processor.prepare_training_data(processed_data)

            if X is None or len(X) == 0:
                return {"error": "Dados insuficientes para avaliação"}

            # Faz predições
            y_pred = self.model.predict(X)

            # Calcula métricas baseadas no tipo de problema
            if self.problem_type == 'classification':
                from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
                accuracy = accuracy_score(y, y_pred)
                report = classification_report(y, y_pred, output_dict=True)
                cm = confusion_matrix(y, y_pred).tolist()

                return {
                    "problem_type": "classification",
                    "accuracy": accuracy,
                    "classification_report": report,
                    "confusion_matrix": cm,
                    "samples_evaluated": len(X)
                }
            else:
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                r2 = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y, y_pred)

                return {
                    "problem_type": "regression",
                    "r2_score": r2,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "samples_evaluated": len(X)
                }

        except Exception as e:
            logger.error(f"Erro na avaliação do modelo: {e}")
            return {"error": str(e)}