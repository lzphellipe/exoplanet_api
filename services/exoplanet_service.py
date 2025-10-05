import requests
import pandas as pd
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ExoplanetService:
    def __init__(self):
        self.base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        self.cache = {}

    def fetch_exoplanet_data(self, query: str = None) -> Optional[List[Dict]]:
        """Busca dados de exoplanetas da API"""
        try:
            if query in self.cache:
                logger.info("Retornando dados do cache")
                return self.cache[query]

            if query is None:
                query = "select+*+from+pscomppars+where+disc_facility+like+%27%25TESS%25%27+order+by+pl_orbper+desc"

            url = f"{self.base_url}?query={query}&format=json"
            logger.info(f"Buscando dados de: {url}")

            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Cache dos dados
            self.cache[query] = data
            logger.info(f"Dados recuperados: {len(data)} registros")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao buscar dados: {e}")
            return None

    def get_tess_exoplanets(self) -> Optional[List[Dict]]:
        """Busca especificamente exoplanetas descobertos pelo TESS"""
        query = "select+*+from+pscomppars+where+disc_facility+like+%27%25TESS%25%27+order+by+pl_orbper+desc"
        return self.fetch_exoplanet_data(query)

    def get_exoplanet_by_name(self, planet_name: str) -> Optional[Dict]:
        """Busca um exoplaneta específico pelo nome"""
        query = f"select+*+from+pscomppars+where+pl_name+like+%27%25{planet_name}%25%27"
        data = self.fetch_exoplanet_data(query)
        return data[0] if data and len(data) > 0 else None

    def get_exoplanets_confirmed(self) -> Optional[Dict]:
        """Busca um exoplaneta específico pelo nome"""
        query = f"select+pl_name,hostname,pl_orbper,pl_bmasse,pl_rade,st_teff,st_mass,st_rad,disc_year,disc_facility+from+ps+where+default_flag=1"
        data = self.fetch_exoplanet_data(query)
        return data[0] if data and len(data) > 0 else None