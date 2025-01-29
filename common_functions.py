import os
import configparser
import random
from datetime import datetime, timedelta
import logging
from faker import Faker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set logging level to DEBUG

fake = Faker('pt_BR')

def load_config(config_path='/app/mount/config.ini'):
    """
    Load configuration from a specified file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        ConfigParser: Loaded configuration object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        Exception: If there is an error loading the configuration.
    """
    logger.debug(f"Attempting to load configuration from: {config_path}")
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        logger.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def table_exists(spark, database_name, table_name):
    """
    Check if a table exists in the Hive Metastore.

    Args:
        spark (SparkSession): The active Spark session.
        database_name (str): The name of the database containing the table.
        table_name (str): The name of the table to check.

    Returns:
        bool: True if the table exists, False otherwise.

    Raises:
        Exception: If there is an error checking the table existence.
    """
    logger.debug(f"Checking existence of table: {database_name}.{table_name}")
    try:
        result = spark.sql(f"SHOW TABLES IN {database_name} LIKE '{table_name}'").count() > 0
        logger.info(f"Table '{database_name}.{table_name}' exists: {result}")
        return result
    except Exception as e:
        logger.error(f"Error checking table existence '{database_name}.{table_name}': {str(e)}")
        raise

def gerar_numero_cartao():
    """
    Generate a random credit card number.

    Returns:
        str: A 16-digit random credit card number.
    """
    logger.debug("Generating credit card number")
    return ''.join([str(random.randint(0, 9)) for _ in range(16)])

def gerar_transacao(clientes_id_usuarios=None):
    """
    Generate a random transaction record.

    Args:
        clientes_id_usuarios (list): Optional list of client user IDs.

    Returns:
        dict: A dictionary containing transaction details.
    """
    logger.debug("Generating transaction record")

    # Calculate the date range for the last 10 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 10)

    return {
        "id_usuario": random.choice(clientes_id_usuarios) if clientes_id_usuarios else random.randint(1, 1000),
        "data_transacao": fake.date_time_between(start_date=start_date, end_date=end_date),
        "valor": round(random.uniform(10, 100000), 2),
        "estabelecimento": fake.company(),
        "categoria": random.choice(["Alimentação", "Transporte", "Entretenimento", "Saúde", "Educação", "Outros"]),
        "status": random.choice(["Aprovada", "Negada", "Pendente", "Estornada", "Cancelada", "Fraude"])
    }

def gerar_cliente():
    """
    Generate a random client record.

    Returns:
        dict: A dictionary containing client details.
    """
    logger.debug("Generating client record")
    ufs = ["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]
    return {
        "id_usuario": random.randint(1, 1000),
        "nome": fake.name(),
        "email": fake.email(),
        "data_nascimento": datetime.strptime(fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat(), "%Y-%m-%d").date(),
        "endereco": fake.address().replace('\n', ', '),
        "limite_credito": random.choice([1000, 2000, 5000, 10000, 20000]),
        "numero_cartao": gerar_numero_cartao(),
        "id_uf": random.choice(ufs)
    }

def gerar_dados(nome_tabela, num_records, clientes_id_usuarios=None):
    """
    Generate random data for a given table.

    Args:
        nome_tabela (str): The name of the table to generate data for.
        num_records (int): The number of records to generate.
        clientes_id_usuarios (list): Optional list of client user IDs.

    Returns:
        list: A list of dictionaries containing generated data.

    Raises:
        ValueError: If the table name is unknown.
    """
    logger.info(f"Generating {num_records} records for table: {nome_tabela}\n")
    if nome_tabela == 'transacoes_cartao':
        return [gerar_transacao(clientes_id_usuarios) for _ in range(num_records)]
    elif nome_tabela == 'clientes':
        return [gerar_cliente() for _ in range(num_records)]
    else:
        logger.error(f"Unknown table: {nome_tabela}")
        raise ValueError(f"Unknown table: {nome_tabela}")
