import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class AIMemory:
    """
    Sistema de memória da IA para armazenar dados de treinamento,
    modelos e histórico de execuções
    """

    def __init__(self):
        self.training_data = None
        self.models_history = []
        self.prediction_history = []
        self.feature_importance_history = []
        self.training_stats = {}

    def store_training_data(self, data: List[Dict]) -> None:
        """Armazena dados de treinamento na memória"""
        try:
            self.training_data = data
            self._calculate_training_stats()
            logger.info(f"Dados de treinamento armazenados: {len(data)} registros")
        except Exception as e:
            logger.error(f"Erro ao armazenar dados de treinamento: {e}")

    def get_training_data(self) -> Optional[List[Dict]]:
        return self.training_data

    def store_model(self, model_info: Dict[str, Any]) -> None:
        try:
            serializable_info = model_info.copy()
            if 'model' in serializable_info:

                pass

            self.models_history.append(serializable_info)

            # Mantém apenas os 10 modelos mais recentes
            if len(self.models_history) > 10:
                self.models_history = self.models_history[-10:]

            logger.info(f"Model stored in memory. Total: {len(self.models_history)}")

        except Exception as e:
            logger.error(f"Error storing model: {e}")

    def get_latest_model(self) -> Optional[Dict[str, Any]]:
        if not self.models_history:
            return None
        return self.models_history[-1]

    def get_model_history(self) -> List[Dict[str, Any]]:
        return self.models_history

    def store_prediction(self, prediction_data: Dict[str, Any]) -> None:
        """Armazena histórico de predições"""
        try:
            prediction_data['timestamp'] = datetime.now().isoformat()
            self.prediction_history.append(prediction_data)

            # Mantém apenas as 1000 predições mais recentes
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]

        except Exception as e:
            logger.error(f"Erro ao armazenar predição: {e}")

    def get_prediction_history(self) -> List[Dict[str, Any]]:
        """Recupera histórico de predições"""
        return self.prediction_history

    def _calculate_training_stats(self) -> None:
        """Calcula estatísticas dos dados de treinamento"""
        try:
            if not self.training_data:
                return

            df = pd.DataFrame(self.training_data)
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            stats = {}
            for col in numeric_cols:
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'count': int(df[col].count()),
                    'null_count': int(df[col].isnull().sum())
                }

            self.training_stats = {
                'overall': {
                    'total_samples': len(df),
                    'total_features': len(df.columns),
                    'numeric_features': len(numeric_cols),
                    'categorical_features': len(df.select_dtypes(include=['object']).columns)
                },
                'column_stats': stats
            }

        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da memória da IA"""
        return {
            "training_data": {
                "has_data": self.training_data is not None,
                "samples": len(self.training_data) if self.training_data else 0
            },
            "models": {
                "total_trained": len(self.models_history),
                "latest_training": self.models_history[-1]['timestamp'] if self.models_history else None
            },
            "predictions": {
                "total_made": len(self.prediction_history)
            },
            "training_stats": self.training_stats
        }

    def clear_memory(self) -> None:
        """Limpa toda a memória da IA"""
        self.training_data = None
        self.models_history = []
        self.prediction_history = []
        self.training_stats = {}
        logger.info("Memória da IA limpa")