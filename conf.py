import os

# Allow overriding connection params via environment variables to keep the app portable
DB_CON_PARAMS = {
    'database': os.getenv('DB_NAME', 'qsweepy'),
    'user': os.getenv('DB_USER', 'qsweepy'),
    'password': os.getenv('DB_PASSWORD', 'qsweepy'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
}

# Connection pool sizing; tuned for light workloads by default.
DB_POOL_MIN = int(os.getenv('DB_POOL_MIN', '1'))
DB_POOL_MAX = int(os.getenv('DB_POOL_MAX', '10'))
# Safety limits to keep UI responsive.
MAX_QUERY_ROWS = int(os.getenv('MAX_QUERY_ROWS', '500'))
STATEMENT_TIMEOUT_MS = int(os.getenv('STATEMENT_TIMEOUT_MS', '10000'))

EXTRACT_QUERIES = "SELECT query_name, query, query_date FROM queries"
DEFAULT_QUERY = "SELECT qubit_id_metadata.value as qubit_id, data.* FROM data\n\
                 LEFT JOIN metadata qubit_id_metadata ON\n\
                 qubit_id_metadata.data_id = data.id AND\n\
                 qubit_id_metadata.name='qubit_id'\n\
                 ORDER BY id DESC;\n\
                 \n"
PATH_test = "/Users/mikhailgoncharov/QtLab/data"
# Where generated SVGs are stored; override via SVG_OUTPUT_PATH env variable.
PATH = os.getenv('SVG_OUTPUT_PATH', "C:\\tupoye-govno")
DEFAULT_QUERY_NAME_PREFIX = "Query from "
