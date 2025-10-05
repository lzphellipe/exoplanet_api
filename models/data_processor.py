import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.problem_type = None
        self.target_column = None

    def process_data(self, raw_data):
        """Processa dados brutos de exoplanetas"""
        try:
            df = pd.DataFrame(raw_data)
            logger.info(f"Colunas disponíveis: {df.columns.tolist()}")
            logger.info(f"Total de registros: {len(df)}")

            # Remove colunas com muitos valores faltantes (>70%)
            threshold = len(df) * 0.7
            df_cleaned = df.dropna(axis=1, thresh=threshold)

            logger.info(f"Colunas após remover missing: {df_cleaned.columns.tolist()}")

            # Seleciona features relevantes para ML - priorizando colunas numéricas importantes
            relevant_features = [
                # Parâmetros orbitais
                'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2',
                'pl_orbsmax', 'pl_orbsmaxerr1', 'pl_orbsmaxerr2',

                # Parâmetros do planeta (se disponíveis)
                'pl_rade', 'pl_radeerr1', 'pl_radeerr2',
                'pl_bmasse', 'pl_bmasseerr1', 'pl_bmasseerr2',
                'pl_dens', 'pl_denserr1', 'pl_denserr2',

                # Temperatura
                'pl_eqt', 'pl_eqterr1', 'pl_eqterr2',

                # Parâmetros da estrela
                'st_teff', 'st_tefferr1', 'st_tefferr2',
                'st_rad', 'st_raderr1', 'st_raderr2',
                'st_mass', 'st_masserr1', 'st_masserr2',
                'st_logg', 'st_loggerr1', 'st_loggerr2',

                # Coordenadas e distância
                'ra', 'dec',
                'sy_dist', 'sy_disterr1', 'sy_disterr2',

                # Informações de descoberta
                'disc_year', 'disc_facility', 'discoverymethod'
            ]

            # Filtra apenas features disponíveis
            available_features = [f for f in relevant_features if f in df_cleaned.columns]
            logger.info(f"Features disponíveis: {available_features}")

            # Se não há features suficientes, usa todas as colunas numéricas
            if len(available_features) < 5:
                numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
                available_features = numeric_cols[:20]  # Limita a 20 colunas numéricas
                logger.info(f"Usando colunas numéricas: {available_features}")

            df_filtered = df_cleaned[available_features]

            # Trata valores numéricos faltantes
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_filtered[col].isna().sum() > 0:
                    df_filtered[col] = df_filtered[col].fillna(df_filtered[col].median())
                    logger.info(f"Preenchidos valores missing em {col}")

            # Codifica variáveis categóricas
            categorical_cols = df_filtered.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in df_filtered.columns:
                    self.label_encoders[col] = LabelEncoder()
                    df_filtered[col] = self.label_encoders[col].fit_transform(
                        df_filtered[col].astype(str)
                    )
                    logger.info(f"Codificada coluna categórica: {col}")

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
            df_copy = df.copy()  # Trabalha em uma cópia para evitar problemas

            # Densidade estelar aproximada
            if 'st_mass' in df_copy.columns and 'st_rad' in df_copy.columns:
                # Evita divisão por zero
                st_rad_safe = df_copy['st_rad'].replace(0, np.nan)
                df_copy['st_density'] = df_copy['st_mass'] / (st_rad_safe ** 3)
                df_copy['st_density'] = df_copy['st_density'].fillna(df_copy['st_density'].median())
                logger.info("Criada feature: st_density")

            # Log do período orbital
            if 'pl_orbper' in df_copy.columns:
                # Adiciona pequeno valor para evitar log(0)
                df_copy['log_orbper'] = np.log10(df_copy['pl_orbper'] + 1e-6)
                logger.info("Criada feature: log_orbper")

            # Ano de descoberta normalizado
            if 'disc_year' in df_copy.columns:
                current_year = 2025
                min_year = df_copy['disc_year'].min()
                max_year = df_copy['disc_year'].max()

                if max_year > min_year:  # Evita divisão por zero
                    df_copy['disc_year_norm'] = (df_copy['disc_year'] - min_year) / (max_year - min_year)
                else:
                    df_copy['disc_year_norm'] = 0.5  # Valor padrão se todos os anos forem iguais
                logger.info("Criada feature: disc_year_norm")

            # Razão raio planeta/estrela (se ambas disponíveis)
            if 'pl_rade' in df_copy.columns and 'st_rad' in df_copy.columns:
                st_rad_safe = df_copy['st_rad'].replace(0, np.nan)
                df_copy['pl_st_ratio'] = df_copy['pl_rade'] / st_rad_safe
                df_copy['pl_st_ratio'] = df_copy['pl_st_ratio'].fillna(df_copy['pl_st_ratio'].median())
                logger.info("Criada feature: pl_st_ratio")

            # Velocidade orbital aproximada
            if 'pl_orbper' in df_copy.columns and 'pl_orbsmax' in df_copy.columns:
                # Filtra períodos válidos (>0)
                valid_mask = (df_copy['pl_orbper'] > 0) & (df_copy['pl_orbsmax'] > 0)
                orbital_velocity = np.zeros(len(df_copy))
                orbital_velocity[valid_mask] = (2 * np.pi * df_copy.loc[valid_mask, 'pl_orbsmax']) / df_copy.loc[
                    valid_mask, 'pl_orbper']
                df_copy['orbital_velocity'] = orbital_velocity
                # Substitui zeros por mediana
                zero_mask = df_copy['orbital_velocity'] == 0
                if zero_mask.any():
                    median_velocity = df_copy.loc[~zero_mask, 'orbital_velocity'].median()
                    df_copy.loc[zero_mask, 'orbital_velocity'] = median_velocity
                logger.info("Criada feature: orbital_velocity")

            logger.info(f"Features derivadas criadas. Colunas: {df_copy.columns.tolist()}")
            return df_copy

        except Exception as e:
            logger.warning(f"Erro ao criar features derivadas: {e}")
            return df  # Retorna o DataFrame original em caso de erro

    def _detect_problem_type(self, y: np.ndarray) -> str:
        """Detecta automaticamente se é classificação ou regressão"""
        # Remove valores NaN
        y_clean = y[~pd.isnull(y)]

        if len(y_clean) == 0:
            return 'regression'  # Default

        unique_values = len(np.unique(y_clean))
        total_samples = len(y_clean)

        # Se há poucos valores únicos em relação ao total, é classificação
        if unique_values / total_samples < 0.1 or unique_values <= 10:
            return 'classification'
        else:
            return 'regression'

    def _create_target_variable(self, df: pd.DataFrame) -> Tuple[np.ndarray, str]:
        """Cria target apropriado baseado nos dados disponíveis"""
        try:
            # Prioridade 1: Raio do planeta (regressão)
            if 'pl_rade' in df.columns and df['pl_rade'].notna().sum() > 10:
                y = df['pl_rade'].values
                target_name = 'pl_rade'
                logger.info(f"Usando pl_rade como target (regressão). Amostras: {df['pl_rade'].notna().sum()}")
                return y, target_name

            # Prioridade 2: Massa do planeta (regressão)
            elif 'pl_bmasse' in df.columns and df['pl_bmasse'].notna().sum() > 10:
                y = df['pl_bmasse'].values
                target_name = 'pl_bmasse'
                logger.info(f"Usando pl_bmasse como target (regressão). Amostras: {df['pl_bmasse'].notna().sum()}")
                return y, target_name

            # Prioridade 3: Período orbital (regressão)
            elif 'pl_orbper' in df.columns and df['pl_orbper'].notna().sum() > 10:
                y = df['pl_orbper'].values
                target_name = 'pl_orbper'
                logger.info(f"Usando pl_orbper como target (regressão). Amostras: {df['pl_orbper'].notna().sum()}")
                return y, target_name

            # Prioridade 4: Classificação por método de descoberta
            elif 'discoverymethod' in df.columns and df['discoverymethod'].notna().sum() > 10:
                # Codifica o método de descoberta como target categórico
                le = LabelEncoder()
                valid_mask = df['discoverymethod'].notna()
                y = np.full(len(df), np.nan)
                y[valid_mask] = le.fit_transform(df.loc[valid_mask, 'discoverymethod'].astype(str))
                target_name = 'discoverymethod'
                logger.info(f"Usando discoverymethod como target (classificação). Classes: {len(le.classes_)}")
                return y, target_name

            # Prioridade 5: Classificação por ano de descoberta (décadas)
            elif 'disc_year' in df.columns and df['disc_year'].notna().sum() > 10:
                # Agrupa anos por décadas para classificação
                valid_mask = df['disc_year'].notna()
                decades = (df.loc[valid_mask, 'disc_year'] // 10) * 10
                y = np.full(len(df), np.nan)
                y[valid_mask] = decades.values
                target_name = 'disc_year_decade'
                logger.info(f"Usando década de descoberta como target (classificação). Décadas: {np.unique(decades)}")
                return y, target_name

            # Fallback: usar primeira coluna numérica disponível
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    y = df[numeric_cols[0]].values
                    target_name = numeric_cols[0]
                    logger.info(
                        f"Fallback: usando {target_name} como target. Amostras: {df[numeric_cols[0]].notna().sum()}")
                    return y, target_name
                else:
                    raise ValueError("Nenhuma coluna adequada encontrada para criar target")

        except Exception as e:
            logger.error(f"Erro ao criar target variable: {e}")
            raise

    def prepare_training_data(self, processed_data: pd.DataFrame,
                              target_col: str = None,
                              problem_type: str = 'auto') -> Tuple[Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[List[str]]]:
        """Prepara dados para treinamento do modelo"""
        try:
            df = processed_data.copy()
            logger.info(f"Preparando dados de treinamento. Shape: {df.shape}")

            # Define a coluna target
            if target_col and target_col in df.columns:
                self.target_column = target_col
                y = df[target_col].values
                X = df.drop(columns=[target_col])
                logger.info(f"Target específico definido: {target_col}")
            else:
                # Cria target automaticamente baseado nos dados disponíveis
                y, self.target_column = self._create_target_variable(df)
                if self.target_column in df.columns:
                    X = df.drop(columns=[self.target_column])
                else:
                    X = df
                logger.info(f"Target automático definido: {self.target_column}")

            # Remove amostras com target missing
            valid_indices = ~pd.isnull(y)
            X = X[valid_indices]
            y = y[valid_indices]

            logger.info(f"Amostras válidas após remover missing: {len(X)}")

            if len(X) == 0:
                raise ValueError("Nenhuma amostra válida após processamento")

            # Detecta o tipo de problema
            if problem_type == 'auto':
                self.problem_type = self._detect_problem_type(y)
            else:
                self.problem_type = problem_type

            logger.info(f"Tipo de problema detectado: {self.problem_type}")
            logger.info(f"Target column: {self.target_column}")
            logger.info(f"Shape dos dados: X{X.shape}, y{y.shape}")

            # Remove colunas com muitos valores faltantes
            X = X.dropna(axis=1, thresh=len(X) * 0.5)  # 50% de threshold
            logger.info(f"Colunas após remover missing: {X.columns.tolist()}")

            # Preenche valores faltantes restantes
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X[col].isna().sum() > 0:
                    X[col] = X[col].fillna(X[col].median())

            # Para classificação, garante que y é inteiro e remove classes raras
            if self.problem_type == 'classification':
                y = y.astype(int)
                # Remove classes com muito poucas amostras
                unique, counts = np.unique(y, return_counts=True)
                valid_classes = unique[counts >= 3]  # Mínimo 3 amostras por classe
                mask = np.isin(y, valid_classes)
                X = X[mask]
                y = y[mask]
                logger.info(f"Classes após filtragem: {np.unique(y)}")
                logger.info(f"Distribuição: {np.bincount(y)}")

            # Normaliza features numéricas
            if len(numeric_cols) > 0:
                X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
                logger.info("Features numéricas normalizadas")

            self.feature_columns = X.columns.tolist()
            logger.info(f"Features para treinamento ({len(self.feature_columns)}): {self.feature_columns}")
            logger.info(f"Total de amostras finais: {len(X)}")

            return X.values, y, self.feature_columns

        except Exception as e:
            logger.error(f"Erro ao preparar dados de treinamento: {e}", exc_info=True)
            return None, None, None

    def get_problem_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o problema de ML"""
        return {
            'problem_type': self.problem_type,
            'target_column': self.target_column,
            'feature_count': len(self.feature_columns),
            'features': self.feature_columns,
            'scaler_fitted': hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None
        }