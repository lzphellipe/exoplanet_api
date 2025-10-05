#!/bin/bash

echo "🚀 Configurando projeto Exoplanet API para deploy..."

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Criar __init__.py em todos os pacotes
echo -e "${YELLOW}📁 Criando __init__.py nos pacotes...${NC}"

# Controllers
cat > controllers/__init__.py << 'EOF'
"""
Controllers package
Contém todos os endpoints da API organizados por domínio
"""
EOF
echo -e "${GREEN}✅ controllers/__init__.py criado${NC}"

# Models
cat > models/__init__.py << 'EOF'
"""
Models package
Contém modelos de dados e processadores
"""
EOF
echo -e "${GREEN}✅ models/__init__.py criado${NC}"

# Services
cat > services/__init__.py << 'EOF'
"""
Services package
Contém serviços de negócio e integrações externas
"""
EOF
echo -e "${GREEN}✅ services/__init__.py criado${NC}"

# Utils
cat > utils/__init__.py << 'EOF'
"""
Utils package
Contém funções utilitárias e helpers
"""
EOF
echo -e "${GREEN}✅ utils/__init__.py criado${NC}"

# Criar Procfile
echo -e "${YELLOW}📝 Criando Procfile...${NC}"
cat > Procfile << 'EOF'
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2 --threads 4 --log-level info
EOF
echo -e "${GREEN}✅ Procfile criado${NC}"

# Criar railway.json
echo -e "${YELLOW}🚂 Criando railway.json...${NC}"
cat > railway.json << 'EOF'
{
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
EOF
echo -e "${GREEN}✅ railway.json criado${NC}"

# Verificar estrutura
echo -e "${YELLOW}🔍 Verificando estrutura do projeto...${NC}"

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅ $1${NC}"
    else
        echo -e "${RED}❌ $1 não encontrado${NC}"
    fi
}

check_file "app.py"
check_file "requirements.txt"
check_file "Procfile"
check_file "railway.json"
check_file "Dockerfile"
check_file ".gitignore"
check_file "controllers/__init__.py"
check_file "models/__init__.py"
check_file "services/__init__.py"
check_file "utils/__init__.py"

echo ""
echo -e "${YELLOW}📋 Próximos passos:${NC}"
echo "1. Revisar e atualizar app.py se necessário"
echo "2. Verificar requirements.txt"
echo "3. git add ."
echo "4. git commit -m 'Preparar para deploy no Railway'"
echo "5. git push origin main"
echo "6. Fazer deploy no Railway"
echo ""
echo -e "${GREEN}✨ Setup completo!${NC}"