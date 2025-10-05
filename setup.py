#!/usr/bin/env python3
"""
Script de setup para preparar o projeto para deploy no Railway
"""
import os
from pathlib import Path


def create_file(filepath, content):
    """Cria um arquivo com o conteúdo especificado"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ {filepath} criado")


def check_file(filepath):
    """Verifica se um arquivo existe"""
    if Path(filepath).exists():
        print(f"✅ {filepath}")
        return True
    else:
        print(f"❌ {filepath} não encontrado")
        return False


def main():
    print("🚀 Configurando projeto Exoplanet API para deploy...\n")

    # Criar __init__.py nos pacotes
    print("📁 Criando __init__.py nos pacotes...")

    init_content = {
        'controllers/__init__.py': '''"""
Controllers package
Contém todos os endpoints da API organizados por domínio
"""
''',
        'models/__init__.py': '''"""
Models package
Contém modelos de dados e processadores
"""
''',
        'services/__init__.py': '''"""
Services package
Contém serviços de negócio e integrações externas
"""
''',
        'utils/__init__.py': '''"""
Utils package
Contém funções utilitárias e helpers
"""
'''
    }

    for filepath, content in init_content.items():
        create_file(filepath, content)

    # Criar Procfile
    print("\n📝 Criando Procfile...")
    procfile_content = "web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2 --threads 4 --log-level info\n"
    create_file('Procfile', procfile_content)

    # Criar railway.json
    print("\n🚂 Criando railway.json...")
    railway_json = '''{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "numReplicas": 1,
    "startCommand": "gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2 --threads 4",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10,
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100
  }
}
'''
    create_file('railway.json', railway_json)

    # Verificar estrutura do projeto
    print("\n🔍 Verificando estrutura do projeto...")
    files_to_check = [
        'app.py',
        'requirements.txt',
        'Procfile',
        'railway.json',
        'Dockerfile',
        '.gitignore',
        'controllers/__init__.py',
        'models/__init__.py',
        'services/__init__.py',
        'utils/__init__.py',
    ]

    all_ok = all(check_file(f) for f in files_to_check)

    # Próximos passos
    print("\n📋 Próximos passos:")
    print("1. Revisar e atualizar app.py se necessário")
    print("2. Verificar requirements.txt")
    print("3. git add .")
    print('4. git commit -m "Preparar para deploy no Railway"')
    print("5. git push origin main")
    print("6. Fazer deploy no Railway")

    if all_ok:
        print("\n✨ Setup completo! Todos os arquivos necessários estão prontos.")
    else:
        print("\n⚠️  Alguns arquivos estão faltando. Revise a lista acima.")


if __name__ == '__main__':
    main()