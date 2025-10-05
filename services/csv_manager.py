import pandas as pd
import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class CSVManager:
    """
    Gerencia o armazenamento de dados em arquivos CSV na pasta data
    com controle de versões e operações de backup
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._ensure_data_directory()

        # Estrutura de arquivos
        self.files = {
            'exoplanets': self.data_dir / "exoplanets.csv",
            'confirmed_planets': self.data_dir / "confirmed_planets.csv",
            'metadata': self.data_dir / "metadata.json",
            'backup_index': self.data_dir / "backup_index.json"
        }

        # Inicializa metadados
        self._initialize_metadata()
        self._initialize_backup_index()

    def _ensure_data_directory(self):
        """Garante que o diretório data existe"""
        try:
            self.data_dir.mkdir(exist_ok=True)
            logger.info(f"Diretório de dados criado/verificado: {self.data_dir.absolute()}")
        except Exception as e:
            logger.error(f"Erro ao criar diretório data: {e}")
            raise

    def _initialize_metadata(self):
        """Inicializa o arquivo de metadados se não existir"""
        if not self.files['metadata'].exists():
            metadata = {
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'files': {},
                'statistics': {
                    'total_operations': 0,
                    'total_records_stored': 0,
                    'last_operation': None
                }
            }
            self._save_metadata(metadata)
            logger.info("Arquivo de metadados inicializado")

    def _initialize_backup_index(self):
        """Inicializa o índice de backups se não existir"""
        if not self.files['backup_index'].exists():
            backup_index = {
                'backups': [],
                'last_backup': None,
                'total_backups': 0
            }
            self._save_backup_index(backup_index)
            logger.info("Índice de backups inicializado")

    def _load_metadata(self) -> Dict:
        """Carrega metadados do arquivo"""
        try:
            if self.files['metadata'].exists():
                with open(self.files['metadata'], 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Erro ao carregar metadados: {e}")
            return {}

    def _save_metadata(self, metadata: Dict):
        """Salva metadados no arquivo"""
        try:
            with open(self.files['metadata'], 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao salvar metadados: {e}")

    def _load_backup_index(self) -> Dict:
        """Carrega índice de backups"""
        try:
            if self.files['backup_index'].exists():
                with open(self.files['backup_index'], 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {'backups': []}
        except Exception as e:
            logger.error(f"Erro ao carregar índice de backups: {e}")
            return {'backups': []}

    def _save_backup_index(self, backup_index: Dict):
        """Salva índice de backups"""
        try:
            with open(self.files['backup_index'], 'w', encoding='utf-8') as f:
                json.dump(backup_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao salvar índice de backups: {e}")

    def _update_file_metadata(self, file_type: str, operation: str, records_count: int, source: str = "api"):
        """Atualiza metadados de um arquivo específico"""
        try:
            metadata = self._load_metadata()

            if 'files' not in metadata:
                metadata['files'] = {}

            if file_type not in metadata['files']:
                metadata['files'][file_type] = {
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'total_operations': 0,
                    'total_records': 0,
                    'operations_history': []
                }

            file_meta = metadata['files'][file_type]
            file_meta['last_updated'] = datetime.now().isoformat()
            file_meta['last_operation'] = operation
            file_meta['last_source'] = source
            file_meta['total_operations'] = file_meta.get('total_operations', 0) + 1

            # Atualiza estatísticas totais
            if operation == 'create':
                file_meta['total_records'] = records_count
            elif operation == 'update':
                file_meta['total_records'] = records_count

            # Adiciona ao histórico de operações
            operation_record = {
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'records_count': records_count,
                'source': source
            }

            if 'operations_history' not in file_meta:
                file_meta['operations_history'] = []

            file_meta['operations_history'].append(operation_record)

            # Mantém apenas as últimas 50 operações no histórico
            if len(file_meta['operations_history']) > 50:
                file_meta['operations_history'] = file_meta['operations_history'][-50:]

            # Atualiza estatísticas gerais
            metadata['last_updated'] = datetime.now().isoformat()
            metadata['statistics']['total_operations'] = metadata['statistics'].get('total_operations', 0) + 1
            metadata['statistics']['last_operation'] = operation_record
            metadata['statistics']['total_records_stored'] = sum(
                f['total_records'] for f in metadata['files'].values()
                if 'total_records' in f
            )

            self._save_metadata(metadata)

        except Exception as e:
            logger.error(f"Erro ao atualizar metadados do arquivo: {e}")

    def save_exoplanets_data(self, data: List[Dict], append: bool = True, source: str = "api") -> Dict[str, Any]:
        """
        Salva dados do endpoint /api/exoplanets no CSV

        Args:
            data: Lista de dicionários com dados dos exoplanetas
            append: Se True, adiciona aos dados existentes
            source: Fonte dos dados ('api', 'sample', 'backup')

        Returns:
            Dict com informações da operação
        """
        try:
            if not data:
                logger.warning("Nenhum dado para salvar em exoplanets.csv")
                return {"success": False, "error": "Nenhum dado fornecido"}

            df = pd.DataFrame(data)
            original_count = len(df)

            # Remove duplicatas baseado no nome do planeta se a coluna existir
            if 'pl_name' in df.columns:
                initial_count = len(df)
                df = df.drop_duplicates(subset=['pl_name'], keep='last')
                duplicates_removed = initial_count - len(df)
                if duplicates_removed > 0:
                    logger.info(f"Removidas {duplicates_removed} duplicatas de exoplanetas")
            else:
                # Remove duplicatas baseado em todas as colunas
                initial_count = len(df)
                df = df.drop_duplicates()
                duplicates_removed = initial_count - len(df)
                if duplicates_removed > 0:
                    logger.info(f"Removidas {duplicates_removed} duplicatas completas")

            # Verifica se o arquivo já existe para fazer append
            file_exists = self.files['exoplanets'].exists()

            if file_exists and append:
                # Carrega dados existentes e combina
                try:
                    existing_df = pd.read_csv(self.files['exoplanets'])

                    # Combina e remove duplicatas
                    if 'pl_name' in df.columns and 'pl_name' in existing_df.columns:
                        combined_df = pd.concat([existing_df, df]).drop_duplicates(
                            subset=['pl_name'],
                            keep='last'
                        )
                    else:
                        combined_df = pd.concat([existing_df, df]).drop_duplicates()

                    operation = "update"
                    records_added = len(combined_df) - len(existing_df)
                    final_count = len(combined_df)

                    combined_df.to_csv(self.files['exoplanets'], index=False)
                    logger.info(f"Dados atualizados. Adicionados: {records_added} registros")

                except Exception as e:
                    logger.error(f"Erro ao ler arquivo existente, criando novo: {e}")
                    df.to_csv(self.files['exoplanets'], index=False)
                    operation = "create"
                    records_added = len(df)
                    final_count = len(df)
            else:
                # Cria novo arquivo
                df.to_csv(self.files['exoplanets'], index=False)
                operation = "create"
                records_added = len(df)
                final_count = len(df)
                logger.info(f"Arquivo criado com {final_count} registros")

            # Atualiza metadados
            self._update_file_metadata('exoplanets', operation, final_count, source)

            result = {
                "success": True,
                "operation": operation,
                "original_records": original_count,
                "records_added": records_added,
                "final_count": final_count,
                "duplicates_removed": duplicates_removed,
                "file_path": str(self.files['exoplanets'])
            }

            logger.info(f"Dados de exoplanetas salvos: {result}")
            return result

        except Exception as e:
            logger.error(f"Erro ao salvar dados de exoplanetas: {e}")
            return {"success": False, "error": str(e)}

    def save_confirmed_planets_data(self, data: List[Dict], append: bool = True, source: str = "api") -> Dict[str, Any]:
        """
        Salva dados do endpoint /confirmed no CSV

        Args:
            data: Lista de dicionários com dados dos planetas confirmados
            append: Se True, adiciona aos dados existentes
            source: Fonte dos dados

        Returns:
            Dict com informações da operação
        """
        try:
            if not data:
                logger.warning("Nenhum dado para salvar em confirmed_planets.csv")
                return {"success": False, "error": "Nenhum dado fornecido"}

            df = pd.DataFrame(data)
            original_count = len(df)

            # Remove duplicatas baseado no nome do planeta
            if 'pl_name' in df.columns:
                initial_count = len(df)
                df = df.drop_duplicates(subset=['pl_name'], keep='last')
                duplicates_removed = initial_count - len(df)
                if duplicates_removed > 0:
                    logger.info(f"Removidas {duplicates_removed} duplicatas de planetas confirmados")

            # Verifica se o arquivo já existe para fazer append
            file_exists = self.files['confirmed_planets'].exists()

            if file_exists and append:
                # Carrega dados existentes e combina
                try:
                    existing_df = pd.read_csv(self.files['confirmed_planets'])

                    # Combina e remove duplicatas
                    if 'pl_name' in df.columns and 'pl_name' in existing_df.columns:
                        combined_df = pd.concat([existing_df, df]).drop_duplicates(
                            subset=['pl_name'],
                            keep='last'
                        )
                    else:
                        combined_df = pd.concat([existing_df, df]).drop_duplicates()

                    operation = "update"
                    records_added = len(combined_df) - len(existing_df)
                    final_count = len(combined_df)

                    combined_df.to_csv(self.files['confirmed_planets'], index=False)
                    logger.info(f"Dados confirmados atualizados. Adicionados: {records_added} registros")

                except Exception as e:
                    logger.error(f"Erro ao ler arquivo existente, criando novo: {e}")
                    df.to_csv(self.files['confirmed_planets'], index=False)
                    operation = "create"
                    records_added = len(df)
                    final_count = len(df)
            else:
                # Cria novo arquivo
                df.to_csv(self.files['confirmed_planets'], index=False)
                operation = "create"
                records_added = len(df)
                final_count = len(df)
                logger.info(f"Arquivo de confirmados criado com {final_count} registros")

            # Atualiza metadados
            self._update_file_metadata('confirmed_planets', operation, final_count, source)

            result = {
                "success": True,
                "operation": operation,
                "original_records": original_count,
                "records_added": records_added,
                "final_count": final_count,
                "duplicates_removed": duplicates_removed,
                "file_path": str(self.files['confirmed_planets'])
            }

            logger.info(f"Dados de planetas confirmados salvos: {result}")
            return result

        except Exception as e:
            logger.error(f"Erro ao salvar dados de planetas confirmados: {e}")
            return {"success": False, "error": str(e)}

    def load_exoplanets_data(self) -> pd.DataFrame:
        """Carrega dados de exoplanetas do CSV"""
        try:
            if self.files['exoplanets'].exists():
                df = pd.read_csv(self.files['exoplanets'])
                logger.info(f"Dados de exoplanetas carregados: {len(df)} registros")
                return df
            else:
                logger.warning("Arquivo exoplanets.csv não encontrado")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao carregar dados de exoplanetas: {e}")
            return pd.DataFrame()

    def load_confirmed_planets_data(self) -> pd.DataFrame:
        """Carrega dados de planetas confirmados do CSV"""
        try:
            if self.files['confirmed_planets'].exists():
                df = pd.read_csv(self.files['confirmed_planets'])
                logger.info(f"Dados de planetas confirmados carregados: {len(df)} registros")
                return df
            else:
                logger.warning("Arquivo confirmed_planets.csv não encontrado")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao carregar dados de planetas confirmados: {e}")
            return pd.DataFrame()

    def get_file_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas detalhadas dos arquivos"""
        stats = {
            'data_directory': str(self.data_dir.absolute()),
            'files': {},
            'summary': {
                'total_files': 0,
                'total_records': 0,
                'total_size_bytes': 0
            }
        }

        for file_type, file_path in self.files.items():
            if file_path.name.endswith('.csv'):
                file_stats = {
                    'exists': file_path.exists(),
                    'path': str(file_path),
                    'type': file_type
                }

                if file_path.exists():
                    try:
                        file_size = file_path.stat().st_size
                        df = pd.read_csv(file_path)
                        record_count = len(df)

                        file_stats.update({
                            'size_bytes': file_size,
                            'size_mb': round(file_size / (1024 * 1024), 2),
                            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                            'records_count': record_count,
                            'columns_count': len(df.columns),
                            'columns': df.columns.tolist()
                        })

                        # Atualiza resumo
                        stats['summary']['total_files'] += 1
                        stats['summary']['total_records'] += record_count
                        stats['summary']['total_size_bytes'] += file_size

                    except Exception as e:
                        logger.error(f"Erro ao analisar arquivo {file_path}: {e}")
                        file_stats['error'] = str(e)
                else:
                    file_stats.update({
                        'size_bytes': 0,
                        'size_mb': 0,
                        'last_modified': None,
                        'records_count': 0,
                        'columns_count': 0,
                        'columns': []
                    })

                stats['files'][file_type] = file_stats

        # Adiciona metadados
        metadata = self._load_metadata()
        stats['metadata'] = metadata

        # Converte tamanho total para MB
        stats['summary']['total_size_mb'] = round(stats['summary']['total_size_bytes'] / (1024 * 1024), 2)

        return stats

    def get_data_summary(self) -> Dict[str, Any]:
        """Retorna resumo completo dos dados armazenados"""
        try:
            exoplanets_df = self.load_exoplanets_data()
            confirmed_df = self.load_confirmed_planets_data()

            summary = {
                'exoplanets': {
                    'total_records': len(exoplanets_df),
                    'columns_count': len(exoplanets_df.columns) if not exoplanets_df.empty else 0,
                    'columns': list(exoplanets_df.columns) if not exoplanets_df.empty else [],
                    'sample_data': exoplanets_df.head(3).to_dict('records') if not exoplanets_df.empty else []
                },
                'confirmed_planets': {
                    'total_records': len(confirmed_df),
                    'columns_count': len(confirmed_df.columns) if not confirmed_df.empty else 0,
                    'columns': list(confirmed_df.columns) if not confirmed_df.empty else [],
                    'sample_data': confirmed_df.head(3).to_dict('records') if not confirmed_df.empty else []
                },
                'storage_info': self.get_file_stats()
            }

            return summary

        except Exception as e:
            logger.error(f"Erro ao gerar resumo de dados: {e}")
            return {}

    def create_backup(self, backup_name: str = None) -> Dict[str, Any]:
        """Cria um backup dos dados atuais"""
        try:
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}"

            backup_dir = self.data_dir / "backups" / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Copia todos os arquivos CSV e de metadados
            files_copied = []
            for file_type, file_path in self.files.items():
                if file_path.exists():
                    backup_file = backup_dir / file_path.name
                    shutil.copy2(file_path, backup_file)
                    files_copied.append(file_path.name)

            # Cria relatório do backup
            backup_report = {
                'name': backup_name,
                'timestamp': datetime.now().isoformat(),
                'files_copied': files_copied,
                'backup_path': str(backup_dir),
                'stats': self.get_file_stats()
            }

            # Salva relatório no backup
            report_file = backup_dir / "backup_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(backup_report, f, indent=2, ensure_ascii=False)

            # Atualiza índice de backups
            backup_index = self._load_backup_index()
            backup_index['backups'].append(backup_report)
            backup_index['last_backup'] = backup_report['timestamp']
            backup_index['total_backups'] = len(backup_index['backups'])

            # Mantém apenas os últimos 10 backups no índice
            if len(backup_index['backups']) > 10:
                backup_index['backups'] = backup_index['backups'][-10:]

            self._save_backup_index(backup_index)

            logger.info(f"Backup criado: {backup_name} com {len(files_copied)} arquivos")

            return {
                "success": True,
                "backup_name": backup_name,
                "files_copied": files_copied,
                "backup_path": str(backup_dir)
            }

        except Exception as e:
            logger.error(f"Erro ao criar backup: {e}")
            return {"success": False, "error": str(e)}

    def list_backups(self) -> List[Dict[str, Any]]:
        """Lista todos os backups disponíveis"""
        try:
            backup_index = self._load_backup_index()
            return backup_index.get('backups', [])
        except Exception as e:
            logger.error(f"Erro ao listar backups: {e}")
            return []

    def restore_backup(self, backup_name: str) -> Dict[str, Any]:
        """Restaura dados de um backup"""
        try:
            backup_dir = self.data_dir / "backups" / backup_name

            if not backup_dir.exists():
                return {"success": False, "error": f"Backup {backup_name} não encontrado"}

            # Verifica se o relatório do backup existe
            report_file = backup_dir / "backup_report.json"
            if not report_file.exists():
                return {"success": False, "error": "Relatório do backup não encontrado"}

            # Cria backup dos dados atuais antes da restauração
            self.create_backup(f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            # Restaura arquivos
            files_restored = []
            for file_path in backup_dir.iterdir():
                if file_path.is_file() and file_path.suffix in ['.csv', '.json']:
                    if file_path.name != "backup_report.json":  # Não restaura o relatório
                        target_file = self.data_dir / file_path.name
                        shutil.copy2(file_path, target_file)
                        files_restored.append(file_path.name)

            logger.info(f"Backup {backup_name} restaurado. Arquivos: {files_restored}")

            return {
                "success": True,
                "backup_name": backup_name,
                "files_restored": files_restored
            }

        except Exception as e:
            logger.error(f"Erro ao restaurar backup: {e}")
            return {"success": False, "error": str(e)}

    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, Any]:
        """Remove dados antigos baseado na data de modificação"""
        try:
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            files_removed = []

            for file_type, file_path in self.files.items():
                if file_path.exists() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    files_removed.append(file_path.name)
                    logger.info(f"Arquivo antigo removido: {file_path.name}")

            return {
                "success": True,
                "files_removed": files_removed,
                "cutoff_days": days_old
            }

        except Exception as e:
            logger.error(f"Erro na limpeza de dados antigos: {e}")
            return {"success": False, "error": str(e)}

    def export_data(self, export_dir: str = "exports") -> Dict[str, Any]:
        """Exporta todos os dados para um diretório externo"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = Path(export_dir) / f"exoplanet_export_{timestamp}"
            export_path.mkdir(parents=True, exist_ok=True)

            # Copia arquivos
            files_exported = []
            for file_type, file_path in self.files.items():
                if file_path.exists():
                    export_file = export_path / file_path.name
                    shutil.copy2(file_path, export_file)
                    files_exported.append(file_path.name)

            # Cria relatório de exportação
            export_report = {
                'export_timestamp': datetime.now().isoformat(),
                'files_exported': files_exported,
                'export_path': str(export_path),
                'stats': self.get_file_stats()
            }

            report_file = export_path / "export_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(export_report, f, indent=2, ensure_ascii=False)

            files_exported.append("export_report.json")

            logger.info(f"Dados exportados para: {export_path}")

            return {
                "success": True,
                "export_path": str(export_path),
                "files_exported": files_exported,
                "total_files": len(files_exported)
            }

        except Exception as e:
            logger.error(f"Erro na exportação: {e}")
            return {"success": False, "error": str(e)}