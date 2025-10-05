"""
Exoplanet Service - Refatorado
Bugs corrigidos e duplicações eliminadas
"""
import requests
import pandas as pd
import logging
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class ExoplanetService:
    """Serviço para buscar dados de exoplanetas da NASA Exoplanet Archive"""

    def __init__(self):
        self.base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        self.cache = {}
        self.timeout = 30

    def fetch_exoplanet_data(self, query: str = None) -> Optional[List[Dict]]:
        """
        Busca dados de exoplanetas da API com tratamento robusto de erros

        Args:
            query: Query ADQL personalizada (opcional)

        Returns:
            Lista de dicionários com dados ou None em caso de erro
        """
        try:
            # Verifica cache
            if query and query in self.cache:
                logger.info("✓ Retornando dados do cache")
                return self.cache[query]

            # Query padrão se não especificada
            if query is None:
                query = (
                    "select * from pscomppars "
                    "where disc_facility like '%TESS%' "
                    "order by pl_orbper desc"
                )

            url = f"{self.base_url}?query={query}&format=json"
            logger.info(f"Buscando dados: {url[:100]}...")

            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Valida resposta
            if not response.content:
                logger.error("Resposta vazia da API")
                return None

            # Parse JSON
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Erro ao decodificar JSON: {e}")
                logger.error(f"Conteúdo: {response.text[:500]}")
                return None

            # Valida tipo de dados
            if not isinstance(data, list):
                logger.error(f"Resposta não é lista: {type(data)}")
                return None

            # Cache
            if query:
                self.cache[query] = data

            logger.info(f"✓ {len(data)} registros recuperados")
            return data

        except requests.exceptions.Timeout:
            logger.error(f"Timeout ao buscar dados (>{self.timeout}s)")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro de requisição: {e}")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado: {e}", exc_info=True)
            return None

    def get_tess_exoplanets(self) -> Optional[List[Dict]]:
        """
        Busca exoplanetas descobertos pelo TESS com fallback

        Returns:
            Lista de dicionários com dados TESS
        """
        try:
            logger.info("Buscando exoplanetas TESS...")

            query = (
                "select * from pscomppars "
                "where disc_facility like '%TESS%' "
                "order by pl_orbper desc"
            )
            data = self.fetch_exoplanet_data(query)

            # Fallback: busca dados gerais se TESS falhar
            if data is None or len(data) == 0:
                logger.warning("Falha ao buscar TESS, usando fallback...")
                fallback_query = "select * from pscomppars limit 100"
                data = self.fetch_exoplanet_data(fallback_query)

            # Último fallback: dados de exemplo
            if data is None or len(data) == 0:
                logger.warning("API falhou completamente, usando dados de exemplo")
                data = self.get_sample_data()

            return data

        except Exception as e:
            logger.error(f"Erro ao buscar exoplanetas TESS: {e}")
            return self.get_sample_data()

    def get_exoplanets_confirmed(self) -> Optional[List[Dict]]:
        """
        Busca exoplanetas confirmados da tabela ps

        Returns:
            Lista de dicionários com planetas confirmados
        """
        try:
            logger.info("Buscando exoplanetas confirmados...")

            query = (
                "select pl_name,hostname,pl_orbper,pl_bmasse,pl_rade,"
                "st_teff,st_mass,st_rad,disc_year,disc_facility,"
                "sy_dist,ra,dec,pl_eqt,pl_dens "
                "from ps "
                "where default_flag=1"
            )

            data = self.fetch_exoplanet_data(query)

            if data and len(data) > 0:
                # Filtra planetas com dados essenciais
                confirmed = [
                    planet for planet in data
                    if planet.get('pl_name') and planet.get('pl_rade') is not None
                ]

                logger.info(f"✓ {len(confirmed)} exoplanetas confirmados com dados válidos")
                return confirmed
            else:
                logger.warning("Nenhum dado confirmado retornado, usando fallback")
                return self.get_tess_exoplanets()

        except Exception as e:
            logger.error(f"Erro ao buscar confirmados: {e}")
            return None

    def fetch_koi_candidates(self) -> Optional[List[Dict]]:
        """
        Busca candidatos KOI (Kepler Objects of Interest)
        Returns:
            Lista de candidatos KOI
        """
        try:

            query = """
                    SELECT *
                    FROM cumulative
                    WHERE koi_disposition IN ('CONFIRMED','FALSE POSITIVE')
                    ORDER BY koi_score DESC \
                    """.replace('\n', ' ').replace('  ', ' ')


            data = self.fetch_exoplanet_data(query)

            if data and len(data) > 0:
                candidates = [
                    koi for koi in data
                    if koi.get('kepoi_name') and koi.get('koi_prad') is not None
                ]

                logger.info(f"✓ {len(candidates)} candidatos KOI válidos")
                return candidates



        except Exception as e:
            logger.error(f"Erro ao buscar candidatos KOI: {e}")
            return None

    def get_exoplanet_by_name(self, planet_name: str) -> Optional[Dict]:
        """
        Busca um exoplaneta específico por nome

        Args:
            planet_name: Nome do exoplaneta

        Returns:
            Dicionário com dados do exoplaneta ou None
        """
        try:
            logger.info(f"Buscando exoplaneta: {planet_name}")

            # Sanitiza nome
            safe_name = planet_name.replace("'", "''")

            query = (
                f"select * from ps "
                f"where pl_name='{safe_name}' "
                f"and default_flag=1"
            )

            data = self.fetch_exoplanet_data(query)

            if data and len(data) > 0:
                logger.info(f"✓ Exoplaneta '{planet_name}' encontrado")
                return data[0]
            else:
                logger.warning(f"Exoplaneta '{planet_name}' não encontrado")
                return None

        except Exception as e:
            logger.error(f"Erro ao buscar {planet_name}: {e}")
            return None

    def get_sample_data(self) -> List[Dict]:
        """
        Retorna dados de exemplo para desenvolvimento

        Returns:
            Lista com dados de exemplo
        """
        logger.info("⚠ Usando dados de exemplo para desenvolvimento")

        return [
            {
                'pl_name': 'TRAPPIST-1 b',
                'hostname': 'TRAPPIST-1',
                'pl_orbper': 1.51087081,
                'pl_bmasse': 1.374,
                'pl_rade': 1.116,
                'st_teff': 2559,
                'st_mass': 0.0898,
                'st_rad': 0.121,
                'disc_year': 2016,
                'discoverymethod': 'Transit',
                'disc_facility': 'TRAPPIST',
                'sy_dist': 12.43,
                'ra': 346.62201,
                'dec': -5.04147
            },
            {
                'pl_name': 'TRAPPIST-1 c',
                'hostname': 'TRAPPIST-1',
                'pl_orbper': 2.4218233,
                'pl_bmasse': 1.308,
                'pl_rade': 1.097,
                'st_teff': 2559,
                'st_mass': 0.0898,
                'st_rad': 0.121,
                'disc_year': 2016,
                'discoverymethod': 'Transit',
                'disc_facility': 'TRAPPIST',
                'sy_dist': 12.43,
                'ra': 346.62201,
                'dec': -5.04147
            },
            {
                'pl_name': 'Kepler-186 f',
                'hostname': 'Kepler-186',
                'pl_orbper': 129.9459,
                'pl_bmasse': None,
                'pl_rade': 1.17,
                'st_teff': 3755,
                'st_mass': 0.544,
                'st_rad': 0.523,
                'disc_year': 2014,
                'discoverymethod': 'Transit',
                'disc_facility': 'Kepler',
                'sy_dist': 178.5,
                'ra': 285.679,
                'dec': 43.9416
            },
            {
                'pl_name': 'Proxima Cen b',
                'hostname': 'Proxima Centauri',
                'pl_orbper': 11.18427,
                'pl_bmasse': 1.07,
                'pl_rade': 1.3,
                'st_teff': 3050,
                'st_mass': 0.122,
                'st_rad': 0.154,
                'disc_year': 2016,
                'discoverymethod': 'Radial Velocity',
                'disc_facility': 'ESO 3.6 m',
                'sy_dist': 1.3,
                'ra': 217.42896,
                'dec': -62.67956
            },
            {
                'pl_name': 'TOI-700 d',
                'hostname': 'TOI-700',
                'pl_orbper': 37.4266,
                'pl_bmasse': None,
                'pl_rade': 1.144,
                'st_teff': 3480,
                'st_mass': 0.415,
                'st_rad': 0.422,
                'disc_year': 2020,
                'discoverymethod': 'Transit',
                'disc_facility': 'TESS',
                'sy_dist': 31.1,
                'ra': 93.0773,
                'dec': -65.4499
            }
        ]

    def search_by_facility(self, facility: str, limit: int = 100) -> Optional[List[Dict]]:
        """
        Busca exoplanetas por instalação de descoberta

        Args:
            facility: Nome da instalação (ex: 'TESS', 'Kepler', 'CHEOPS')
            limit: Limite de resultados

        Returns:
            Lista de exoplanetas descobertos pela instalação
        """
        try:
            query = (
                f"select top {limit} * from ps "
                f"where disc_facility like '%{facility}%' "
                f"and default_flag=1"
            )

            data = self.fetch_exoplanet_data(query)

            if data:
                logger.info(f"✓ {len(data)} exoplanetas de {facility}")

            return data

        except Exception as e:
            logger.error(f"Erro ao buscar por facility '{facility}': {e}")
            return None

    def get_statistics(self) -> Dict:
        """
        Retorna estatísticas sobre o cache e uso do serviço

        Returns:
            Dicionário com estatísticas
        """
        return {
            'cache_size': len(self.cache),
            'cached_queries': list(self.cache.keys()),
            'base_url': self.base_url,
            'timeout': self.timeout
        }

    def clear_cache(self):
        """Limpa o cache de queries"""
        self.cache.clear()
        logger.info("✓ Cache limpo")