import pandas as pd
import lightkurve as lk
import numpy as np
import os
import logging
from src.core.config import config

logger = logging.getLogger(__name__)


class LightkurveService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(os.path.join(config.data_dir, "lightcurves"), exist_ok=True)
        self.features_file = os.path.join(config.data_dir, "lightkurve_features.csv")
        self.lightcurve_dir = os.path.join(config.data_dir, "lightcurves")

    def fetch_lightcurve(self, hostname: str, mission: str = "TESS", max_attempts: int = 3) -> lk.LightCurve:
        """Busca a curva de luz para um hostname (estrela hospedeira)."""
        self.logger.info(f"Buscando curva de luz para hostname {hostname} (missão: {mission})")
        for attempt in range(max_attempts):
            try:
                search_result = lk.search_lightcurve(hostname, mission=mission)
                if len(search_result) == 0:
                    self.logger.warning(f"Nenhuma curva de luz encontrada para {hostname}")
                    return None
                # Baixar a primeira curva de luz disponível
                lightcurve = search_result[0].download()
                # Salvar a curva de luz como arquivo .fits
                file_path = os.path.join(self.lightcurve_dir, f"{hostname.replace(' ', '_')}_lightcurve.fits")
                lightcurve.to_fits(file_path, overwrite=True)
                self.logger.info(f"Curva de luz salva em {file_path}")
                return lightcurve
            except Exception as e:
                self.logger.error(f"Erro na tentativa {attempt + 1} para {hostname}: {e}")
                if attempt == max_attempts - 1:
                    self.logger.error(f"Falha ao buscar curva de luz para {hostname}")
                    return None
        return None

    def extract_features(self, lightcurve: lk.LightCurve) -> dict:
        """Extrai features de uma curva de luz."""
        if lightcurve is None:
            return {
                "transit_depth": None,
                "transit_duration": None,
                "period": None
            }

        try:
            # Normalizar a curva de luz
            lightcurve = lightcurve.normalize()
            # Encontrar períodos candidatos usando periodograma
            periodogram = lightcurve.to_periodogram(method="bls", minimum_period=0.5, maximum_period=100)
            period = periodogram.period_at_max_power.value
            # Converter para folded lightcurve para extrair trânsitos
            folded = lightcurve.fold(period=period)
            transit_mask = folded.get_transit_mask(period=period)
            transit_depth = np.abs(folded.flux[transit_mask].min() - 1.0) if transit_mask.any() else None
            transit_duration = folded.time[transit_mask].ptp() * period if transit_mask.any() else None

            self.logger.info(
                f"Features extraídas: período={period:.2f} dias, profundidade={transit_depth}, duração={transit_duration}")
            return {
                "transit_depth": transit_depth,
                "transit_duration": transit_duration,
                "period": period
            }
        except Exception as e:
            self.logger.error(f"Erro ao extrair features: {e}")
            return {
                "transit_depth": None,
                "transit_duration": None,
                "period": None
            }

    def process_dataset(self, dataset: str, max_samples: int = None) -> pd.DataFrame:
        """Processa curvas de luz para um dataset e salva features."""
        self.logger.info(f"Processando curvas de luz para dataset: {dataset}")
        file_path = f"{config.data_dir}/{dataset}_data.json"

        try:
            df = pd.read_json(file_path)
            if max_samples:
                df = df.head(max_samples)

            features_list = []
            for idx, row in df.iterrows():
                target = row.get("tic_id", row.get("pl_name", None))
                if not target:
                    self.logger.warning(f"Sem TIC ID ou pl_name para índice {idx}")
                    continue

                lightcurve = self.fetch_lightcurve(target)
                features = self.extract_features(lightcurve)
                features["pl_name"] = row["pl_name"]
                features_list.append(features)

            features_df = pd.DataFrame(features_list)
            features_df.to_csv(self.features_file, index=False)
            self.logger.info(f"Features salvas em {self.features_file} com {len(features_df)} registros")
            return features_df
        except Exception as e:
            self.logger.error(f"Erro ao processar dataset {dataset}: {e}")
            raise