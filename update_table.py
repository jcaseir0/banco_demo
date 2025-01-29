import os
import json
import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.functions import current_date
from common_functions import load_config, gerar_dados, table_exists

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_table(spark, table_name, partition_by=None):
    """
    Update a table with new data, optionally partitioning by a specified column.

    Args:
        spark (SparkSession): The active Spark session.
        table_name (str): The name of the table to update.
        partition_by (str, optional): The column to partition by, if any.

    Raises:
        Exception: If an error occurs during the update process.
    """
    logger.info(f"Updating table: {table_name}")
    logger.debug(f"Partition by: {partition_by}")
    try:
        if partition_by:
            logger.debug(f"Inserting data with partition: {partition_by}")
            spark.sql(f"""
                INSERT INTO {table_name}
                PARTITION ({partition_by}=current_date())
                SELECT * FROM temp_view
            """)
        else:
            logger.debug("Inserting data without partition")
            spark.sql(f"INSERT INTO {table_name} SELECT * FROM temp_view")
        logger.info(f"Data inserted into table '{table_name}' successfully.")
    except Exception as e:
        logger.error(f"Error updating table '{table_name}': {str(e)}")
        raise

def get_schema_path(base_path, table_name):
    """
    Generate the schema path for a given table.
    
    Args:
    base_path (str): The base directory path where schema files are stored.
    table_name (str): The name of the table.
    
    Returns:
    str: The full path to the schema file for the given table.
    """
    schema_filename = f"{table_name}.json"
    return os.path.join(base_path, "schemas", schema_filename)

def main():
    """
    Main function to update tables based on configuration.

    This function loads the configuration, iterates through the tables,
    generates new data, and updates each existing table.
    """
    logger.info("Starting table update process")
    spark = SparkSession.builder.appName("UpdateTable").getOrCreate()
    logger.debug("Spark session created")

    config = load_config()
    logger.debug("Configuration loaded")

    # Acessando a lista de tabelas diretamente da seção DEFAULT
    for table_name in config.get("DEFAULT", "tables").split(","):
        table_name = table_name.strip()  # Remove espaços em branco se houver
        logger.info(f"Processing table: {table_name}")

        # Acessando as configurações da tabela usando o nome da tabela
        num_records = config.getint(table_name, 'num_records', fallback=100)
        partition_by = config.get(table_name, 'partition_by', fallback=None)
        base_path = "/app/mount"
        schema_path = get_schema_path(base_path, table_name)

        logger.debug(f"Schema path: {schema_path}")
        logger.debug(f"Number of records: {num_records}")
        logger.debug(f"Partition by: {partition_by}")

        if not os.path.exists(schema_path):
            logger.error(f"Schema file not found for table '{table_name}': {schema_path}")
            continue

        with open(schema_path, 'r') as f:
            schema = json.load(f)
        logger.debug("Schema loaded")

        database_name = config.get("DEFAULT", "dbname")
        if table_exists(spark, database_name, table_name):
            logger.info(f"Updating existing table: {table_name}")
            data = gerar_dados(table_name, num_records)
            df = spark.createDataFrame(data, schema=StructType.fromJson(schema))
            df = df.withColumn("data_execucao", current_date())
            df.createOrReplaceTempView("temp_view")
            update_table(spark, table_name, partition_by)
        else:
            logger.warning(f"Table '{table_name}' does not exist. Cannot update.")

    logger.info("Table update process completed")
    spark.stop()
    logger.debug("Spark session stopped")

if __name__ == "__main__":
    main()
