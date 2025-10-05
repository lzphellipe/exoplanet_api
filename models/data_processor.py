import pandas as pd
import numpy as np
from flask import config
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []

    def process_data(self, raw_data):
        """Processa dados brutos de exoplanetas"""
        try:
            df = pd.DataFrame(raw_data)
            logger.info(f"Colunas disponíveis: {df.columns.tolist()}")

            # Remove colunas com muitos valores faltantes
            threshold = len(df) * 0.7
            df_cleaned = df.dropna(axis=1, thresh=threshold)

            # Seleciona features relevantes para ML
            relevant_features = [
                'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2',  # Período orbital
                'pl_orbsmax', 'pl_orbsmaxerr1', 'pl_orbsmaxerr2',  # Semi-eixo maior
                'pl_rade', 'pl_radeerr1', 'pl_radeerr2',  # Raio do planeta
                'pl_bmasse', 'pl_bmasseerr1', 'pl_bmasseerr2',  # Massa do planeta
                'pl_dens', 'pl_denserr1', 'pl_denserr2',  # Densidade
                'pl_eqt', 'pl_eqterr1', 'pl_eqterr2',  # Temperatura de equilíbrio
                'st_teff', 'st_tefferr1', 'st_tefferr2',  # Temperatura da estrela
                'st_rad', 'st_raderr1', 'st_raderr2',  # Raio da estrela
                'st_mass', 'st_masserr1', 'st_masserr2',  # Massa da estrela
                'st_logg', 'st_loggerr1', 'st_loggerr2',  # Gravidade da estrela
                'ra', 'dec',  # Coordenadas
                'sy_dist', 'sy_disterr1', 'sy_disterr2',  # Distância
                'disc_year', 'disc_facility'  # Informações de descoberta
            ]

            # Filtra apenas features disponíveis
            available_features = [f for f in relevant_features if f in df_cleaned.columns]
            df_filtered = df_cleaned[available_features]

            # Trata valores numéricos faltantes
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_filtered.loc[:, col] = df_filtered[col].fillna(df_filtered[col].median())

            # Codifica variáveis categóricas
            categorical_cols = df_filtered.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in df_filtered.columns:
                    self.label_encoders[col] = LabelEncoder()
                    df_filtered[col] = self.label_encoders[col].fit_transform(
                        df_filtered[col].astype(str)
                    )

            # Adiciona features derivadas
            df_processed = self._create_derived_features(df_filtered)

            logger.info(f"Dados processados: {df_processed.shape}")
            return df_processed

        except Exception as e:
            logger.error(f"Erro no processamento de dados: {e}")
            raise

    def _create_derived_features(self, df):
        """Cria features derivadas para melhorar o modelo"""
        try:
            # Densidade estelar aproximada
            if 'st_mass' in df.columns and 'st_rad' in df.columns:
                df['st_density'] = df['st_mass'] / (df['st_rad'] ** 3)

            # Razão raio planeta/estrela
            if 'pl_rade' in df.columns and 'st_rad' in df.columns:
                df['pl_st_ratio'] = df['pl_rade'] / df['st_rad']

            # Velocidade orbital aproximada
            if 'pl_orbper' in df.columns and 'pl_orbsmax' in df.columns:
                df['orbital_velocity'] = (2 * np.pi * df['pl_orbsmax']) / df['pl_orbper']

            # Fluxo estelar recebido pelo planeta
            if 'st_teff' in df.columns and 'pl_orbsmax' in df.columns:
                df['received_flux'] = (df['st_teff'] ** 4) / (df['pl_orbsmax'] ** 2)

            return df

        except Exception as e:
            logger.warning(f"Erro ao criar features derivadas: {e}")
            return df

    def prepare_training_data(self, processed_data, target_col='pl_rade'):
        """Prepara dados para treinamento do modelo"""
        try:
            df = processed_data.copy()

            # Remove colunas alvo e colunas com muitos NaN
            if target_col in df.columns:
                y = df[target_col]
                X = df.drop(columns=[target_col])
            else:
                # Se não há target, usa classificação binária baseada no raio
                median_radius = df['pl_rade'].median() if 'pl_rade' in df.columns else 0
                y = (df['pl_rade'] > median_radius).astype(int) if 'pl_rade' in df.columns else pd.Series([0] * len(df))
                X = df

            # Remove colunas com muitos valores faltantes
            X = X.dropna(axis=1, thresh=len(X) * 0.8)

            # Preenche valores faltantes restantes
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                X[col] = X[col].fillna(X[col].median())

            # Normaliza features numéricas
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])

            self.feature_columns = X.columns.tolist()
            logger.info(f"Features para treinamento: {self.feature_columns}")

            return X.values, y.values, self.feature_columns

        except Exception as e:
            logger.error(f"Erro ao preparar dados de treinamento: {e}")
            return None, None, None

    def load_and_process(self, dataset: str) -> pd.DataFrame:
        """Carrega e processa dados de um arquivo JSON."""
        file_path = f"{config.data_dir}/{dataset}_data.json"
        self.logger.info(f"Carregando dados de {file_path}")

        try:
            df = pd.read_json(file_path)
            # Definir colunas específicas para o dataset
            columns = ['pl_orbper', 'pl_bmasse', 'pl_rade', 'st_teff', 'st_mass',
                       'st_rad'] if dataset == 'confirmed' else df.columns.tolist()
            return self.preprocess_data(df, columns)
        except Exception as e:
            self.logger.error(f"Erro ao carregar/processar {file_path}: {e}")
            raise