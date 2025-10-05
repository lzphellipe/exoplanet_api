"""
Lightkurve Service
Processa curvas de luz de exoplanetas
"""
import pandas as pd
import lightkurve as lk
import numpy as np
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """Configuração  para lightkurve service"""

    def __init__(self):
        # Diretório de dados
        self.data_dir = os.path.join(os.getcwd(), "data")

        # Cria diretórios necessários
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "lightcurves"), exist_ok=True)

        logger.info(f"Diretório de dados: {self.data_dir}")


# Instância global de configuração
config = Config()


class LightkurveService:
    """Serviço para buscar e processar light curves usando Lightkurve"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Garante que diretórios existem
        self.data_dir = config.data_dir
        self.lightcurve_dir = os.path.join(self.data_dir, "lightcurves")
        self.features_file = os.path.join(self.data_dir, "lightkurve_features.csv")

        # Cria diretórios se não existirem
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.lightcurve_dir, exist_ok=True)

        self.logger.info(f"LightkurveService inicializado")
        self.logger.info(f"  - Data dir: {self.data_dir}")
        self.logger.info(f"  - Lightcurve dir: {self.lightcurve_dir}")

    def search_lightcurves(self, target_name: str, mission: str = "TESS",
                           quarter: int = None, campaign: int = None) -> dict:
        """
        Busca light curves disponíveis para um alvo

        Args:
            target_name: Nome do alvo (planeta ou estrela)
            mission: Missão (TESS, Kepler, K2)
            quarter: Quarter específico (Kepler)
            campaign: Campaign específico (K2)

        Returns:
            Dict com resultados da busca ou erro
        """
        try:
            self.logger.info(f"Buscando light curves para {target_name} (missão: {mission})")

            # Busca light curves
            search_result = lk.search_lightcurve(target_name, mission=mission)

            if len(search_result) == 0:
                return {
                    "error": f"Nenhuma light curve encontrada para {target_name}",
                    "target": target_name,
                    "mission": mission
                }

            # Filtra por quarter/campaign se especificado
            if quarter is not None:
                search_result = search_result[search_result.quarter == quarter]
            if campaign is not None:
                search_result = search_result[search_result.campaign == campaign]

            if len(search_result) == 0:
                return {
                    "error": "Nenhuma light curve encontrada com os filtros especificados",
                    "target": target_name,
                    "mission": mission,
                    "quarter": quarter,
                    "campaign": campaign
                }

            # Converte resultado para lista de dicionários
            results = []
            for i, item in enumerate(search_result):
                result_info = {
                    "index": i,
                    "target_name": target_name,
                    "mission": str(item.mission),
                    "author": str(item.author) if hasattr(item, 'author') else None,
                    "exptime": str(item.exptime) if hasattr(item, 'exptime') else None,
                    "distance": str(item.distance) if hasattr(item, 'distance') else None
                }

                # Adiciona quarter ou campaign
                if hasattr(item, 'quarter'):
                    result_info['quarter'] = int(item.quarter)
                if hasattr(item, 'campaign'):
                    result_info['campaign'] = int(item.campaign)

                results.append(result_info)

            self.logger.info(f"✓ {len(results)} light curve(s) encontrada(s)")

            return {
                "target": target_name,
                "mission": mission,
                "count": len(results),
                "results": results
            }

        except Exception as e:
            self.logger.error(f"Erro ao buscar light curves: {e}")
            return {
                "error": str(e),
                "target": target_name,
                "mission": mission
            }

    def download_lightcurve(self, target_name: str, mission: str = "TESS",
                            index: int = 0, quarter: int = None,
                            campaign: int = None) -> dict:
        """
        Faz download de uma light curve específica

        Args:
            target_name: Nome do alvo
            mission: Missão
            index: Índice da light curve nos resultados da busca
            quarter: Quarter específico
            campaign: Campaign específico

        Returns:
            Dict com dados da light curve ou erro
        """
        try:
            self.logger.info(f"Baixando light curve {index} para {target_name}")

            # Busca light curves
            search_result = lk.search_lightcurve(target_name, mission=mission)

            if len(search_result) == 0:
                return {
                    "error": f"Nenhuma light curve encontrada para {target_name}",
                    "target": target_name
                }

            # Filtra se necessário
            if quarter is not None:
                search_result = search_result[search_result.quarter == quarter]
            if campaign is not None:
                search_result = search_result[search_result.campaign == campaign]

            if index >= len(search_result):
                return {
                    "error": f"Índice {index} fora do range (0-{len(search_result) - 1})",
                    "target": target_name
                }

            # Download da light curve
            lc = search_result[index].download()

            # Salva como arquivo FITS
            safe_filename = target_name.replace(" ", "_").replace("/", "_")
            fits_path = os.path.join(
                self.lightcurve_dir,
                f"{safe_filename}_lc_{index}.fits"
            )
            lc.to_fits(fits_path, overwrite=True)

            # Extrai dados básicos
            time = lc.time.value.tolist() if hasattr(lc, 'time') else []
            flux = lc.flux.value.tolist() if hasattr(lc, 'flux') else []

            # Limita a 1000 pontos para não sobrecarregar resposta
            if len(time) > 1000:
                step = len(time) // 1000
                time = time[::step]
                flux = flux[::step]

            self.logger.info(f"✓ Light curve salva em {fits_path}")

            return {
                "target": target_name,
                "mission": mission,
                "index": index,
                "file_path": fits_path,
                "data_points": len(lc.time) if hasattr(lc, 'time') else 0,
                "time_sample": time[:100],  # Primeiros 100 pontos
                "flux_sample": flux[:100],
                "statistics": {
                    "mean_flux": float(np.mean(lc.flux.value)) if hasattr(lc, 'flux') else None,
                    "std_flux": float(np.std(lc.flux.value)) if hasattr(lc, 'flux') else None,
                    "min_flux": float(np.min(lc.flux.value)) if hasattr(lc, 'flux') else None,
                    "max_flux": float(np.max(lc.flux.value)) if hasattr(lc, 'flux') else None
                }
            }

        except Exception as e:
            self.logger.error(f"Erro ao baixar light curve: {e}")
            return {
                "error": str(e),
                "target": target_name,
                "mission": mission
            }

    def fetch_lightcurve(self, hostname: str, mission: str = "TESS",
                         max_attempts: int = 3) -> lk.LightCurve:
        """
        Busca a curva de luz para um hostname (estrela hospedeira)

        Args:
            hostname: Nome da estrela hospedeira
            mission: Missão (TESS, Kepler, K2)
            max_attempts: Número máximo de tentativas

        Returns:
            Objeto LightCurve ou None
        """
        self.logger.info(f"Buscando curva de luz para hostname {hostname} (missão: {mission})")

        for attempt in range(max_attempts):
            try:
                search_result = lk.search_lightcurve(hostname, mission=mission)

                if len(search_result) == 0:
                    self.logger.warning(f"Nenhuma curva de luz encontrada para {hostname}")
                    return None

                # Baixa a primeira curva de luz disponível
                lightcurve = search_result[0].download()

                # Salva a curva de luz como arquivo .fits
                safe_filename = hostname.replace(' ', '_').replace('/', '_')
                file_path = os.path.join(
                    self.lightcurve_dir,
                    f"{safe_filename}_lightcurve.fits"
                )
                lightcurve.to_fits(file_path, overwrite=True)

                self.logger.info(f"✓ Curva de luz salva em {file_path}")
                return lightcurve

            except Exception as e:
                self.logger.error(f"Erro na tentativa {attempt + 1} para {hostname}: {e}")
                if attempt == max_attempts - 1:
                    self.logger.error(f"Falha ao buscar curva de luz para {hostname}")
                    return None

        return None

    def extract_features(self, lightcurve: lk.LightCurve) -> dict:
        """
        Extrai features de uma curva de luz

        Args:
            lightcurve: Objeto LightCurve

        Returns:
            Dict com features extraídas
        """
        if lightcurve is None:
            return {
                "transit_depth": None,
                "transit_duration": None,
                "period": None
            }

        try:
            # Normaliza a curva de luz
            lightcurve = lightcurve.normalize()

            # Encontra períodos candidatos usando periodograma
            periodogram = lightcurve.to_periodogram(
                method="bls",
                minimum_period=0.5,
                maximum_period=100
            )
            period = periodogram.period_at_max_power.value

            # Converte para folded lightcurve para extrair trânsitos
            folded = lightcurve.fold(period=period)

            # Detecta trânsitos
            try:
                transit_mask = folded.get_transit_mask(period=period)
                transit_depth = (
                    np.abs(folded.flux[transit_mask].min() - 1.0)
                    if transit_mask.any() else None
                )
                transit_duration = (
                    folded.time[transit_mask].ptp() * period
                    if transit_mask.any() else None
                )
            except Exception as e:
                self.logger.warning(f"Erro ao detectar trânsitos: {e}")
                transit_depth = None
                transit_duration = None

            self.logger.info(
                f"Features extraídas: período={period:.2f} dias, "
                f"profundidade={transit_depth}, duração={transit_duration}"
            )

            return {
                "transit_depth": float(transit_depth) if transit_depth is not None else None,
                "transit_duration": float(transit_duration) if transit_duration is not None else None,
                "period": float(period)
            }

        except Exception as e:
            self.logger.error(f"Erro ao extrair features: {e}")
            return {
                "transit_depth": None,
                "transit_duration": None,
                "period": None,
                "error": str(e)
            }

    def process_dataset(self, dataset: str, max_samples: int = None) -> pd.DataFrame:
        """
        Processa curvas de luz para um dataset e salva features

        Args:
            dataset: Nome do dataset ('confirmed', 'tess', etc)
            max_samples: Número máximo de amostras a processar

        Returns:
            DataFrame com features extraídas
        """
        self.logger.info(f"Processando curvas de luz para dataset: {dataset}")

        file_path = os.path.join(self.data_dir, f"{dataset}_data.json")

        # Verifica se arquivo existe
        if not os.path.exists(file_path):
            # Tenta CSV como alternativa
            csv_path = os.path.join(self.data_dir, f"{dataset}_planets.csv")
            if os.path.exists(csv_path):
                file_path = csv_path
            else:
                self.logger.error(f"Arquivo {file_path} não encontrado")
                return pd.DataFrame()

        try:
            # Carrega dados
            if file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                df = pd.read_csv(file_path)

            if max_samples:
                df = df.head(max_samples)

            features_list = []

            for idx, row in df.iterrows():
                # Tenta diferentes colunas para identificar o alvo
                target = (
                        row.get("tic_id") or
                        row.get("hostname") or
                        row.get("pl_name") or
                        None
                )

                if not target:
                    self.logger.warning(f"Sem identificador para índice {idx}")
                    continue

                # Busca curva de luz
                lightcurve = self.fetch_lightcurve(str(target))

                # Extrai features
                features = self.extract_features(lightcurve)
                features["pl_name"] = row.get("pl_name", target)
                features["target"] = target

                features_list.append(features)

            # Cria DataFrame com features
            features_df = pd.DataFrame(features_list)

            # Salva em CSV
            features_df.to_csv(self.features_file, index=False)

            self.logger.info(
                f"✓ Features salvas em {self.features_file} "
                f"com {len(features_df)} registros"
            )

            return features_df

        except Exception as e:
            self.logger.error(f"Erro ao processar dataset {dataset}: {e}")
            raise

    def get_statistics(self) -> dict:
        """
        Retorna estatísticas sobre light curves processadas

        Returns:
            Dict com estatísticas
        """
        try:
            stats = {
                "lightcurve_dir": self.lightcurve_dir,
                "features_file": self.features_file,
                "lightcurves_saved": 0,
                "features_available": False
            }

            # Conta light curves salvas
            if os.path.exists(self.lightcurve_dir):
                fits_files = list(Path(self.lightcurve_dir).glob("*.fits"))
                stats["lightcurves_saved"] = len(fits_files)

            # Verifica se features existem
            if os.path.exists(self.features_file):
                stats["features_available"] = True
                df = pd.read_csv(self.features_file)
                stats["features_count"] = len(df)
                stats["features_columns"] = df.columns.tolist()

            return stats

        except Exception as e:
            self.logger.error(f"Erro ao obter estatísticas: {e}")
            return {"error": str(e)}