import re


def validate_parameters(params, required_params):
    """Valida parâmetros da requisição"""
    missing_params = [param for param in required_params if param not in params]
    if missing_params:
        return False, f"Parâmetros faltantes: {', '.join(missing_params)}"
    return True, ""


def sanitize_target_name(target_name):
    """Sanitiza nome do alvo para busca no LightKurve"""
    # Remove caracteres especiais e espaços extras
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', target_name).strip()
    return sanitized


def calculate_statistics(data):
    """Calcula estatísticas básicas dos dados"""
    if not data:
        return {}

    df = pd.DataFrame(data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median())
        }

    return stats