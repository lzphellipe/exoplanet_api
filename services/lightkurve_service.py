"""
Lightkurve Service - COM RETRY AUTOMÁTICO
Trata erros de transação do MAST Archive
"""
import pandas as pd
import lightkurve as lk
import numpy as np
import os
import logging
from pathlib import Path
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class Config:
    """Configuração local para lightkurve service"""
    def __init__(self):
        self.data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "lightcurves"), exist_ok=True)
        logger.info(f"Diretório de dados: {self.data_dir}")


config = Config()


class LightkurveService:
    """Serviço para buscar light curves com retry automático"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_dir = config.data_dir
        self.lightcurve_dir = os.path.join(self.data_dir, "lightcurves")
        self.features_file = os.path.join(self.data_dir, "lightkurve_features.csv")

        # Configurações de retry
        self.max_retries = 3
        self.retry_delay = 2  # segundos
        self.backoff_factor = 2  # multiplicador do delay a cada retry

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.lightcurve_dir, exist_ok=True)

        self.logger.info(f"LightkurveService inicializado")
        self.logger.info(f"  - Data dir: {self.data_dir}")
        self.logger.info(f"  - Max retries: {self.max_retries}")

    def _validate_target_mission(self, target_name: str, mission: str) -> Dict[str, Any]:
        """
        Valida se o target é compatível com a missão

        Args:
            target_name: Nome do alvo
            mission: Missão (TESS, Kepler, K2)

        Returns:
            Dict com validação e sugestões
        """
        target_lower = target_name.lower()
        mission_upper = mission.upper()

        warnings = []
        suggestions = []

        # Detecta missão pelo nome
        if 'kepler' in target_lower or 'koi' in target_lower or 'kic' in target_lower:
            expected_mission = 'Kepler'
            if mission_upper == 'TESS':
                warnings.append(f"'{target_name}' parece ser um alvo Kepler, mas missão especificada é TESS")
                suggestions.append(f"Tente mission='Kepler' ou mission='K2'")

        elif 'toi' in target_lower or 'tic' in target_lower:
            expected_mission = 'TESS'
            if mission_upper != 'TESS':
                warnings.append(f"'{target_name}' parece ser um alvo TESS, mas missão especificada é {mission}")
                suggestions.append(f"Tente mission='TESS'")

        return {
            'valid': len(warnings) == 0,
            'warnings': warnings,
            'suggestions': suggestions
        }

    def _retry_operation(self, operation, *args, **kwargs):
        """
        Executa operação com retry automático

        Args:
            operation: Função a executar
            *args, **kwargs: Argumentos para a função

        Returns:
            Resultado da operação ou None se todas as tentativas falharem
        """
        last_error = None
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                result = operation(*args, **kwargs)

                if attempt > 0:
                    self.logger.info(f"✓ Sucesso na tentativa {attempt + 1}")

                return result

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()

                # Verifica se é erro de transação SQL (retry-able)
                is_transaction_error = any([
                    'transaction' in error_msg,
                    'availability replica' in error_msg,
                    'ghost records' in error_msg,
                    'snapshot isolation' in error_msg
                ])

                # Verifica se é erro de timeout (retry-able)
                is_timeout_error = any([
                    'timeout' in error_msg,
                    'timed out' in error_msg
                ])

                # Verifica se é erro de conexão (retry-able)
                is_connection_error = any([
                    'connection' in error_msg,
                    'network' in error_msg
                ])

                should_retry = is_transaction_error or is_timeout_error or is_connection_error

                if should_retry and attempt < self.max_retries - 1:
                    self.logger.warning(
                        f"⚠️ Tentativa {attempt + 1}/{self.max_retries} falhou: {e}"
                    )
                    self.logger.info(f"⏳ Aguardando {delay}s antes de retentar...")
                    time.sleep(delay)
                    delay *= self.backoff_factor  # Aumenta delay exponencialmente
                else:
                    # Última tentativa ou erro não retry-able
                    if should_retry:
                        self.logger.error(
                            f"✗ Todas as {self.max_retries} tentativas falharam"
                        )
                    else:
                        self.logger.error(f"✗ Erro não retry-able: {e}")
                    break

        return None

    def search_lightcurves(self, target_name: str, mission: str = "TESS",
                          quarter: int = None, campaign: int = None) -> dict:
        """
        Busca light curves COM RETRY automático

        Args:
            target_name: Nome do alvo
            mission: Missão (TESS, Kepler, K2)
            quarter: Quarter (Kepler)
            campaign: Campaign (K2)

        Returns:
            Dict com resultados ou erro
        """
        try:
            self.logger.info(f"Buscando light curves para {target_name} (missão: {mission})")

            # Valida target vs missão
            validation = self._validate_target_mission(target_name, mission)
            if not validation['valid']:
                self.logger.warning(f"Validação: {validation['warnings'][0]}")

            # Busca com retry
            def search_operation():
                return lk.search_lightcurve(target_name, mission=mission)

            search_result = self._retry_operation(search_operation)

            if search_result is None:
                return {
                    "error": f"Falha ao buscar light curves após {self.max_retries} tentativas",
                    "target": target_name,
                    "mission": mission,
                    "validation": validation if not validation['valid'] else None
                }

            if len(search_result) == 0:
                error_response = {
                    "error": f"Nenhuma light curve encontrada para {target_name}",
                    "target": target_name,
                    "mission": mission
                }

                # Adiciona sugestões se houver problema de missão
                if not validation['valid']:
                    error_response['suggestions'] = validation['suggestions']

                return error_response

            # Filtra por quarter/campaign se especificado
            if quarter is not None:
                search_result = search_result[search_result.quarter == quarter]
            if campaign is not None:
                search_result = search_result[search_result.campaign == campaign]

            if len(search_result) == 0:
                return {
                    "error": "Nenhuma light curve encontrada com os filtros",
                    "target": target_name,
                    "mission": mission,
                    "quarter": quarter,
                    "campaign": campaign
                }

            # Converte resultado
            results = []
            for i, item in enumerate(search_result):
                result_info = {
                    "index": i,
                    "target_name": target_name,
                    "mission": str(item.mission),
                    "author": str(item.author) if hasattr(item, 'author') else None,
                    "exptime": str(item.exptime) if hasattr(item, 'exptime') else None,
                }

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
                "results": results,
                "validation": validation if not validation['valid'] else None
            }

        except Exception as e:
            self.logger.error(f"Erro ao buscar light curves: {e}", exc_info=True)
            return {
                "error": str(e),
                "target": target_name,
                "mission": mission,
                "error_type": type(e).__name__
            }

    def download_lightcurve(self, target_name: str, mission: str = "TESS",
                           index: int = 0, quarter: int = None,
                           campaign: int = None) -> dict:
        """
        Download de light curve COM RETRY automático

        Args:
            target_name: Nome do alvo
            mission: Missão
            index: Índice da light curve
            quarter: Quarter
            campaign: Campaign

        Returns:
            Dict com dados ou erro
        """
        try:
            self.logger.info(f"Baixando light curve {index} para {target_name}")

            # Busca com retry
            def search_operation():
                return lk.search_lightcurve(target_name, mission=mission)

            search_result = self._retry_operation(search_operation)

            if search_result is None:
                return {
                    "error": f"Falha ao buscar após {self.max_retries} tentativas",
                    "target": target_name
                }

            if len(search_result) == 0:
                return {
                    "error": f"Nenhuma light curve encontrada",
                    "target": target_name
                }

            # Filtra
            if quarter is not None:
                search_result = search_result[search_result.quarter == quarter]
            if campaign is not None:
                search_result = search_result[search_result.campaign == campaign]

            if index >= len(search_result):
                return {
                    "error": f"Índice {index} fora do range (0-{len(search_result)-1})",
                    "target": target_name
                }

            # Download com retry
            def download_operation():
                return search_result[index].download()

            lc = self._retry_operation(download_operation)

            if lc is None:
                return {
                    "error": "Falha no download após múltiplas tentativas",
                    "target": target_name
                }

            # Salva
            safe_filename = target_name.replace(" ", "_").replace("/", "_")
            fits_path = os.path.join(
                self.lightcurve_dir,
                f"{safe_filename}_lc_{index}.fits"
            )
            lc.to_fits(fits_path, overwrite=True)

            # Extrai dados
            time_data = lc.time.value.tolist() if hasattr(lc, 'time') else []
            flux_data = lc.flux.value.tolist() if hasattr(lc, 'flux') else []

            # Limita pontos
            if len(time_data) > 1000:
                step = len(time_data) // 1000
                time_data = time_data[::step]
                flux_data = flux_data[::step]

            self.logger.info(f"✓ Light curve salva em {fits_path}")

            return {
                "target": target_name,
                "mission": mission,
                "index": index,
                "file_path": fits_path,
                "data_points": len(lc.time) if hasattr(lc, 'time') else 0,
                "time_sample": time_data[:100],
                "flux_sample": flux_data[:100],
                "statistics": {
                    "mean_flux": float(np.mean(lc.flux.value)) if hasattr(lc, 'flux') else None,
                    "std_flux": float(np.std(lc.flux.value)) if hasattr(lc, 'flux') else None,
                    "min_flux": float(np.min(lc.flux.value)) if hasattr(lc, 'flux') else None,
                    "max_flux": float(np.max(lc.flux.value)) if hasattr(lc, 'flux') else None
                }
            }

        except Exception as e:
            self.logger.error(f"Erro ao baixar light curve: {e}", exc_info=True)
            return {
                "error": str(e),
                "target": target_name,
                "mission": mission,
                "error_type": type(e).__name__
            }

    def fetch_lightcurve(self, hostname: str, mission: str = "TESS",
                        max_attempts: int = None) -> Optional[lk.LightCurve]:
        """
        Busca light curve COM RETRY

        Args:
            hostname: Nome da estrela
            mission: Missão
            max_attempts: Número de tentativas (usa self.max_retries se None)

        Returns:
            LightCurve ou None
        """
        if max_attempts is None:
            max_attempts = self.max_retries

        self.logger.info(f"Buscando curva de luz para {hostname} (missão: {mission})")

        def fetch_operation():
            search_result = lk.search_lightcurve(hostname, mission=mission)

            if len(search_result) == 0:
                return None

            lightcurve = search_result[0].download()

            # Salva
            safe_filename = hostname.replace(' ', '_').replace('/', '_')
            file_path = os.path.join(
                self.lightcurve_dir,
                f"{safe_filename}_lightcurve.fits"
            )
            lightcurve.to_fits(file_path, overwrite=True)

            self.logger.info(f"✓ Curva de luz salva em {file_path}")
            return lightcurve

        return self._retry_operation(fetch_operation)

    def extract_features(self, lightcurve: lk.LightCurve) -> dict:
        """Extrai features de uma light curve"""
        if lightcurve is None:
            return {
                "transit_depth": None,
                "transit_duration": None,
                "period": None
            }

        try:
            lightcurve = lightcurve.normalize()
            periodogram = lightcurve.to_periodogram(
                method="bls",
                minimum_period=0.5,
                maximum_period=100
            )
            period = periodogram.period_at_max_power.value
            folded = lightcurve.fold(period=period)

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
            except:
                transit_depth = None
                transit_duration = None

            return {
                "transit_depth": float(transit_depth) if transit_depth else None,
                "transit_duration": float(transit_duration) if transit_duration else None,
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
        """Processa dataset com retry automático"""
        self.logger.info(f"Processando dataset: {dataset}")

        file_path = os.path.join(self.data_dir, f"{dataset}_data.json")

        if not os.path.exists(file_path):
            csv_path = os.path.join(self.data_dir, f"{dataset}_planets.csv")
            if os.path.exists(csv_path):
                file_path = csv_path
            else:
                self.logger.error(f"Arquivo não encontrado: {file_path}")
                return pd.DataFrame()

        try:
            if file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                df = pd.read_csv(file_path)

            if max_samples:
                df = df.head(max_samples)

            features_list = []

            for idx, row in df.iterrows():
                target = (
                    row.get("tic_id") or
                    row.get("hostname") or
                    row.get("pl_name") or
                    None
                )

                if not target:
                    continue

                lightcurve = self.fetch_lightcurve(str(target))
                features = self.extract_features(lightcurve)
                features["pl_name"] = row.get("pl_name", target)
                features["target"] = target

                features_list.append(features)

            features_df = pd.DataFrame(features_list)
            features_df.to_csv(self.features_file, index=False)

            self.logger.info(f"✓ {len(features_df)} features salvas")

            return features_df

        except Exception as e:
            self.logger.error(f"Erro: {e}")
            raise

    def get_statistics(self) -> dict:
        """Retorna estatísticas"""
        try:
            stats = {
                "lightcurve_dir": self.lightcurve_dir,
                "features_file": self.features_file,
                "lightcurves_saved": 0,
                "features_available": False,
                "retry_config": {
                    "max_retries": self.max_retries,
                    "retry_delay": self.retry_delay,
                    "backoff_factor": self.backoff_factor
                }
            }

            if os.path.exists(self.lightcurve_dir):
                fits_files = list(Path(self.lightcurve_dir).glob("*.fits"))
                stats["lightcurves_saved"] = len(fits_files)

            if os.path.exists(self.features_file):
                stats["features_available"] = True
                df = pd.read_csv(self.features_file)
                stats["features_count"] = len(df)
                stats["features_columns"] = df.columns.tolist()

            return stats

        except Exception as e:
            self.logger.error(f"Erro: {e}")
            return {"error": str(e)}