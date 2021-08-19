DB_CON_PARAMS = {'database':'qsweepy', 'user':'qsweepy', 'password':'qsweepy'}
EXTRACT_QUERIES = "SELECT query_name, query, query_date FROM queries"
DEFAULT_QUERY = "SELECT qubit_id_metadata.value as qubit_id, data.* FROM data\n\
                 LEFT JOIN metadata qubit_id_metadata ON\n\
                 qubit_id_metadata.data_id = data.id AND\n\
                 qubit_id_metadata.name=\'qubit_id\'\n\
                 ORDER BY id DESC;\n\
                 \n"
PATH_test = "/Users/mikhailgoncharov/QtLab/data"
PATH = "C:\\tupoye-govno"
DEFAULT_QUERY_NAME_PREFIX = "Query from "
