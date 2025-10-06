"""
CSV Manager - Refatorado
Eliminadas duplicações em save_data
"""
import pandas as pd
import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class CSVManager:
    """
    Gerencia armazenamento de dados em CSV com controle de versões
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._ensure_data_directory()

        # Estrutura de arquivos
        self.files = {
            'exoplanets': self.data_dir / "exoplanets.csv",
            'confirmed_planets': self.data_dir / "confirmed_planets.csv",
            'koi_candidates': self.data_dir / "koi_candidates.csv",
            'metadata': self.data_dir / "metadata.json",
            'backup_index': self.data_dir / "backup_index.json"
        }

        self._initialize_metadata()
        self._initialize_backup_index()

    def _ensure_data_directory(self):
        """Garante que o diretório data existe"""
        try:
            self.data_dir.mkdir(exist_ok=True)
            logger.info(f"Diretório de dados: {self.data_dir.absolute()}")
        except Exception as e:
            logger.error(f"Erro ao criar diretório data: {e}")
            raise

    def _initialize_metadata(self):
        """Inicializa arquivo de metadados"""
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

    def _initialize_backup_index(self):
        """Inicializa índice de backups"""
        if not self.files['backup_index'].exists():
            backup_index = {
                'backups': [],
                'last_backup': None,
                'total_backups': 0
            }
            self._save_backup_index(backup_index)

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

    def _update_file_metadata(self, file_type: str, operation: str,
                              records_count: int, source: str = "api"):
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

            if operation in ['create', 'update']:
                file_meta['total_records'] = records_count

            operation_record = {
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'records_count': records_count,
                'source': source
            }

            if 'operations_history' not in file_meta:
                file_meta['operations_history'] = []

            file_meta['operations_history'].append(operation_record)

            # Mantém apenas últimas 50 operações
            if len(file_meta['operations_history']) > 50:
                file_meta['operations_history'] = file_meta['operations_history'][-50:]

            metadata['last_updated'] = datetime.now().isoformat()
            metadata['statistics']['total_operations'] = \
                metadata['statistics'].get('total_operations', 0) + 1
            metadata['statistics']['last_operation'] = operation_record
            metadata['statistics']['total_records_stored'] = sum(
                f['total_records'] for f in metadata['files'].values()
                if 'total_records' in f
            )

            self._save_metadata(metadata)

        except Exception as e:
            logger.error(f"Erro ao atualizar metadados: {e}")

    def save_data(self, data: List[Dict], file_key: str,
                  append: bool = True, source: str = "api",
                  unique_column: str = 'pl_name') -> Dict[str, Any]:

        try:
            if not data:
                logger.warning(f"Nenhum dado para salvar em {file_key}")
                return {"success": False, "error": "Nenhum dado fornecido"}

            if file_key not in self.files:
                logger.error(f"file_key inválido: {file_key}")
                return {"success": False, "error": f"file_key '{file_key}' não existe"}

            file_path = self.files[file_key]
            df = pd.DataFrame(data)
            original_count = len(df)


            duplicates_removed = 0
            if unique_column in df.columns:
                initial_count = len(df)
                df = df.drop_duplicates(subset=[unique_column], keep='last')
                duplicates_removed = initial_count - len(df)
                if duplicates_removed > 0:
                    logger.info(f"Removidas {duplicates_removed} duplicatas em {unique_column}")
            else:
                initial_count = len(df)
                df = df.drop_duplicates()
                duplicates_removed = initial_count - len(df)

            file_exists = file_path.exists()

            if file_exists and append:
                try:
                    existing_df = pd.read_csv(file_path)

                    # Combina e remove duplicatas
                    if unique_column in df.columns and unique_column in existing_df.columns:
                        combined_df = pd.concat([existing_df, df]).drop_duplicates(
                            subset=[unique_column],
                            keep='last'
                        )
                    else:
                        combined_df = pd.concat([existing_df, df]).drop_duplicates()

                    operation = "update"
                    records_added = len(combined_df) - len(existing_df)
                    final_count = len(combined_df)

                    combined_df.to_csv(file_path, index=False)
                    logger.info(f"{file_key}: +{records_added} registros (total: {final_count})")

                except Exception as e:
                    logger.error(f"Erro ao ler arquivo existente: {e}")
                    df.to_csv(file_path, index=False)
                    operation = "create"
                    records_added = len(df)
                    final_count = len(df)
            else:
                df.to_csv(file_path, index=False)
                operation = "create"
                records_added = len(df)
                final_count = len(df)
                logger.info(f"{file_key}: arquivo criado com {final_count} registros")

            self._update_file_metadata(file_key, operation, final_count, source)

            return {
                "success": True,
                "operation": operation,
                "original_records": original_count,
                "records_added": records_added,
                "final_count": final_count,
                "duplicates_removed": duplicates_removed,
                "file_path": str(file_path)
            }

        except Exception as e:
            logger.error(f"Erro ao salvar dados em {file_key}: {e}")
            return {"success": False, "error": str(e)}

    # Métodos de conveniência (wrappers)
    def save_exoplanets_data(self, data: List[Dict], append: bool = True,
                             source: str = "api") -> Dict[str, Any]:
        """Salva dados de exoplanetas"""
        return self.save_data(data, 'exoplanets', append, source)

    def save_confirmed_planets_data(self, data: List[Dict], append: bool = True,
                                    source: str = "api") -> Dict[str, Any]:
        return self.save_data(data, 'confirmed_planets', append, source)

    def save_koi_candidates(self, data: List[Dict], append: bool = True,
                            source: str = "api") -> Dict[str, Any]:
        return self.save_data(data, 'koi_candidates', append, source,
                              unique_column='kepoi_name')

    def load_data(self, file_key: str) -> pd.DataFrame:

        try:
            if file_key not in self.files:
                logger.error(f"file_key invalid: {file_key}")
                return pd.DataFrame()

            file_path = self.files[file_key]

            if file_path.exists():
                df = pd.read_csv(file_path)
                logger.info(f"{file_key}: upload {len(df)} rows")
                return df
            else:
                logger.warning(f"Archive {file_key} não encontrado")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading {file_key}: {e}")
            return pd.DataFrame()

    # Métodos de conveniência para load
    def load_exoplanets_data(self) -> pd.DataFrame:
        return self.load_data('exoplanets')

    def load_confirmed_planets_data(self) -> pd.DataFrame:
        return self.load_data('confirmed_planets')

    def load_koi_candidates(self) -> pd.DataFrame:
        return self.load_data('koi_candidates')

    def get_file_stats(self) -> Dict[str, Any]:
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
                            'last_modified': datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            ).isoformat(),
                            'records_count': record_count,
                            'columns_count': len(df.columns),
                            'columns': df.columns.tolist()
                        })

                        stats['summary']['total_files'] += 1
                        stats['summary']['total_records'] += record_count
                        stats['summary']['total_size_bytes'] += file_size

                    except Exception as e:
                        logger.error(f"Erro ao analisar {file_path}: {e}")
                        file_stats['error'] = str(e)

                stats['files'][file_type] = file_stats

        metadata = self._load_metadata()
        stats['metadata'] = metadata
        stats['summary']['total_size_mb'] = round(
            stats['summary']['total_size_bytes'] / (1024 * 1024), 2
        )

        return stats

    def get_data_summary(self) -> Dict[str, Any]:
        try:
            summary = {}

            for file_key in ['exoplanets', 'confirmed_planets', 'koi_candidates']:
                df = self.load_data(file_key)
                summary[file_key] = {
                    'total_records': len(df),
                    'columns_count': len(df.columns) if not df.empty else 0,
                    'columns': list(df.columns) if not df.empty else [],
                    'sample_data': df.head(3).to_dict('records') if not df.empty else []
                }

            summary['storage_info'] = self.get_file_stats()
            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {}

    def create_backup(self, backup_name: str = None) -> Dict[str, Any]:
        try:
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}"

            backup_dir = self.data_dir / "backups" / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)

            files_copied = []
            for file_type, file_path in self.files.items():
                if file_path.exists():
                    backup_file = backup_dir / file_path.name
                    shutil.copy2(file_path, backup_file)
                    files_copied.append(file_path.name)

            backup_report = {
                'name': backup_name,
                'timestamp': datetime.now().isoformat(),
                'files_copied': files_copied,
                'backup_path': str(backup_dir),
                'stats': self.get_file_stats()
            }

            report_file = backup_dir / "backup_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(backup_report, f, indent=2, ensure_ascii=False)

            backup_index = self._load_backup_index()
            backup_index['backups'].append(backup_report)
            backup_index['last_backup'] = backup_report['timestamp']
            backup_index['total_backups'] = len(backup_index['backups'])

            if len(backup_index['backups']) > 10:
                backup_index['backups'] = backup_index['backups'][-10:]

            self._save_backup_index(backup_index)

            logger.info(f"✓ Backup '{backup_name}' created {len(files_copied)} archives")

            return {
                "success": True,
                "backup_name": backup_name,
                "files_copied": files_copied,
                "backup_path": str(backup_dir)
            }

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return {"success": False, "error": str(e)}

    def list_backups(self) -> List[Dict[str, Any]]:
        try:
            backup_index = self._load_backup_index()
            return backup_index.get('backups', [])
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []

    def restore_backup(self, backup_name: str) -> Dict[str, Any]:
        try:
            backup_dir = self.data_dir / "backups" / backup_name

            if not backup_dir.exists():
                return {"success": False, "error": f"Backup '{backup_name}' not found"}

            self.create_backup(f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            files_restored = []
            for file_path in backup_dir.iterdir():
                if file_path.is_file() and file_path.suffix in ['.csv', '.json']:
                    if file_path.name != "backup_report.json":
                        target_file = self.data_dir / file_path.name
                        shutil.copy2(file_path, target_file)
                        files_restored.append(file_path.name)

            logger.info(f"✓ Backup '{backup_name}' restored")

            return {
                "success": True,
                "backup_name": backup_name,
                "files_restored": files_restored
            }

        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return {"success": False, "error": str(e)}