import requests
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
import json
from services.csv_manager import CSVManager

logger = logging.getLogger(__name__)


class ExoplanetService:
    def __init__(self):
        self.base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        self.cache = {}
        self.csv_manager = CSVManager()

    def fetch_exoplanet_data(self, query: str = None) -> Optional[List[Dict]]:
        """Busca dados de exoplanetas da API com tratamento robusto de erros"""
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

            # Verifica se a resposta não está vazia
            if not response.content:
                logger.error("Resposta vazia da API")
                return None

            # Tenta fazer parse do JSON
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Erro ao decodificar JSON: {e}")
                logger.error(f"Conteúdo da resposta: {response.text[:500]}")  # Log dos primeiros 500 chars
                return None

            # Verifica se os dados são uma lista
            if not isinstance(data, list):
                logger.error(f"Resposta não é uma lista: {type(data)}")
                return None

            # Cache dos dados
            self.cache[query] = data
            logger.info(f"Dados recuperados: {len(data)} registros")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Erro de requisição ao buscar dados: {e}")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado ao buscar dados: {e}")
            return None

    def get_tess_exoplanets(self) -> Optional[List[Dict]]:
        """Busca especificamente exoplanetas descobertos pelo TESS com fallback"""
        try:
            query = "select+*+from+pscomppars+where+disc_facility+like+%27%25TESS%25%27+order+by+pl_orbper+desc"
            data = self.fetch_exoplanet_data(query)

            if data is None:
                logger.warning("Falha ao buscar dados do TESS, tentando fallback...")
                # Fallback: busca dados gerais
                fallback_query = "select+*+from+pscomppars+limit+100"
                data = self.fetch_exoplanet_data(fallback_query)

            if data:
                # Tenta salvar no CSV mesmo com dados limitados
                try:
                    self.csv_manager.save_exoplanets_data(data)
                except Exception as e:
                    logger.warning(f"Erro ao salvar no CSV: {e}")

            return data

        except Exception as e:
            logger.error(f"Erro ao buscar exoplanetas do TESS: {e}")
            return None

    def get_exoplanets_confirmed(self) -> Optional[List[Dict]]:
        """
        Busca exoplanetas confirmados da tabela ps com dados essenciais
        e salva automaticamente no CSV
        """
        try:
            query = (
                "select+pl_name,hostname,pl_orbper,pl_bmasse,pl_rade,st_teff,st_mass,st_rad,"
                "disc_year,disc_facility,sy_dist,ra,dec,pl_eqt,pl_dens+from+ps+where+default_flag=1"
            )
            data = self.fetch_exoplanet_data(query)

            if data:
                logger.info(f"Encontrados {len(data)} exoplanetas confirmados")

                # Filtra planetas com dados essenciais
                confirmed_planets = [
                    planet for planet in data
                    if planet.get('pl_name') and planet.get('pl_rade') is not None
                ]

                # Salva no CSV
                try:
                    commit_id = self.csv_manager.save_confirmed_planets(confirmed_planets, append=True)
                    logger.info(f"Dados confirmados salvos no CSV. Commit: {commit_id}")
                except Exception as e:
                    logger.warning(f"Erro ao salvar dados confirmados no CSV: {e}")

                return confirmed_planets
            else:
                logger.warning("Nenhum dado retornado para exoplanetas confirmados")
                # Tenta usar dados do cache ou fallback
                return self.get_tess_exoplanets()

        except Exception as e:
            logger.error(f"Erro ao buscar exoplanetas confirmados: {e}")
            return None

    def get_sample_data(self) -> List[Dict]:
        """Retorna dados de exemplo para desenvolvimento quando a API falha"""
        logger.info("Usando dados de exemplo para desenvolvimento")

        sample_data = [
            {
                'pl_name': 'TRAPPIST-1 b',
                'pl_orbper': 1.51087081,
                'pl_bmasse': 1.374,
                'pl_rade': 1.116,
                'st_teff': 2559,
                'st_mass': 0.0898,
                'st_rad': 0.121,
                'disc_year': 2016,
                'discoverymethod': 'Transit',
                'disc_facility': 'TRAPPIST'
            },
            {
                'pl_name': 'TRAPPIST-1 c',
                'pl_orbper': 2.4218233,
                'pl_bmasse': 1.308,
                'pl_rade': 1.097,
                'st_teff': 2559,
                'st_mass': 0.0898,
                'st_rad': 0.121,
                'disc_year': 2016,
                'discoverymethod': 'Transit',
                'disc_facility': 'TRAPPIST'
            },
            {
                'pl_name': 'Kepler-186 f',
                'pl_orbper': 129.9459,
                'pl_bmasse': None,
                'pl_rade': 1.17,
                'st_teff': 3755,
                'st_mass': 0.544,
                'st_rad': 0.523,
                'disc_year': 2014,
                'discoverymethod': 'Transit',
                'disc_facility': 'Kepler'
            }
        ]

        return sample_data