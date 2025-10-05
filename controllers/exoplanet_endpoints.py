import json
import os

import requests
from flask import jsonify, request, config
import logging

import app
from services.exoplanet_service import ExoplanetService
from models.ai_memory import AIMemory

logger = logging.getLogger(__name__)

exoplanet_service = ExoplanetService()
ai_memory = AIMemory()


def configure_exoplanet_endpoints(app):
    """Configura os endpoints de exoplanetas"""

    @app.route('/api/exoplanets', methods=['GET'])
    def get_exoplanets():
        """Endpoint para buscar dados brutos de exoplanetas"""
        try:
            data = exoplanet_service.get_tess_exoplanets()
            if data is None:
                return jsonify({"error": "Falha ao buscar dados"}), 500

            # Armazena na memória da IA
            ai_memory.store_training_data(data)

            return jsonify({
                "message": "Dados recuperados e armazenados na memória da IA",
                "count": len(data),
                "memory_stats": ai_memory.get_memory_stats()
            })
        except Exception as e:
            logger.error(f"Erro no endpoint /api/exoplanets: {e}")
            return jsonify({"error": str(e)}), 500



    @app.route('/api/exoplanets/<planet_name>', methods=['GET'])
    def get_exoplanet(planet_name):
        """Endpoint para buscar um exoplaneta específico"""
        try:
            planet = exoplanet_service.get_exoplanet_by_name(planet_name)
            if planet is None:
                return jsonify({"error": f"Exoplaneta {planet_name} não encontrado"}), 404

            return jsonify({
                "message": "Exoplaneta encontrado",
                "data": planet
            })
        except Exception as e:
            logger.error(f"Erro no endpoint /api/exoplanets/{planet_name}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/memory/stats', methods=['GET'])
    def get_memory_stats():
        """Endpoint para estatísticas da memória da IA"""
        try:
            stats = ai_memory.get_memory_stats()
            return jsonify({
                "message": "Estatísticas da memória da IA",
                "stats": stats
            })
        except Exception as e:
            logger.error(f"Erro no endpoint /api/memory/stats: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/memory/clear', methods=['POST'])
    def clear_memory():
        """Endpoint para limpar a memória da IA"""
        try:
            ai_memory.clear_memory()
            return jsonify({
                "message": "Memória da IA limpa com sucesso"
            })
        except Exception as e:
            logger.error(f"Erro no endpoint /api/memory/clear: {e}")
            return jsonify({"error": str(e)}), 500



@app.route('/api/confirmed', methods=['GET'])
def fetch_confirmed(self):
        """Busca dados da API de exoplanetas confirmados e salva em JSON."""
        self.logger.info("Iniciando fetch de dados confirmados")
        try:
           confirmed = exoplanet_service.get_exoplanets_confirmed()
           if confirmed is None:
               return jsonify({"error": "Falha ao buscar dados"}),404
           return jsonify({
               "message": "Confirmado com sucesso",
           })
        except Exception as e:
            logger.error(f"Erro no endpoint /api/confirmed: {e}")
            return jsonify({"error": str(e)}), 500

