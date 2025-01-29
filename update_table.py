import os
import json
import logging
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import StructType
from pyspark.sql.functions import lit
from common_functions import load_config, gerar_dados, table_exists
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_table(spark, database_name, table_name, partition_by=None):
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

    current_date = spark.sql("SELECT CURRENT_DATE() as date").collect()[0]['date']
    try:
        if partition_by:
            logger.debug(f"Inserting data with partition: {partition_by}")
            spark.sql(f"""
                INSERT INTO {database_name}.{table_name}
                PARTITION ({partition_by}='{current_date}')
                SELECT * FROM temp_view
            """)
        else:
            logger.debug("Inserting data without partition")
            spark.sql(f"INSERT INTO {database_name}.{table_name} SELECT * FROM temp_view")
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
    config = load_config()
    logger.debug("Configuration loaded")
    jdbc_url = config['DEFAULT'].get('hmsUrl')
    thrift_server = config['DEFAULT'].get('thriftServer')

    logger.debug(f"JDBC URL: {jdbc_url}")
    logger.debug(f"Thrift Server: {thrift_server}")

    spark_conf = SparkConf()
    spark_conf.set("hive.metastore.client.factory.class", "com.cloudera.spark.hive.metastore.HivemetastoreClientFactory")
    spark_conf.set("hive.metastore.uris", thrift_server)
    spark_conf.set("spark.sql.hive.metastore.jars", "builtin")
    spark_conf.set("spark.sql.hive.hiveserver2.jdbc.url", jdbc_url)

    spark = SparkSession.builder.config(conf=spark_conf).appName("UpdateTable").enableHiveSupport().getOrCreate()
    logger.debug("Spark session created")

    database_name = config.get("DEFAULT", "dbname")
    tables = config.get("DEFAULT", "tables").split(",")
    logger.debug(f"Tables to process: {tables}")
    base_path = "/app/mount"

    # Generate clientes data first
    clientes_table = [table for table in tables if 'clientes' in table][0]
    logger.debug(f"Clientes table: {clientes_table}")
    clientes_num_records = config.getint(clientes_table, 'num_records', fallback=100)
    clientes_data = gerar_dados(clientes_table, clientes_num_records)
    clientes_id_usuarios = [cliente['id_usuario'] for cliente in clientes_data]

    # Acessando a lista de tabelas diretamente da seção DEFAULT
    for table_name in tables:
        table_name = table_name.strip()  # Remove espaços em branco se houver
        logger.info(f"Processing table: {table_name}")

        # Acessando as configurações da tabela usando o nome da tabela
        num_records_update = config.getint(table_name, 'num_records_update', fallback=100)
        partition_by = config.get(table_name, 'partition_by', fallback=None)
        schema_path = get_schema_path(base_path, table_name)

        logger.debug(f"Schema path: {schema_path}")
        logger.debug(f"Number of records: {num_records_update}")
        logger.debug(f"Partition by: {partition_by}")

        if not os.path.exists(schema_path):
            logger.error(f"Schema file not found for table '{table_name}': {schema_path}")
            continue

        with open(schema_path, 'r') as f:
            schema = json.load(f)
        logger.debug("Schema loaded")

        if table_exists(spark, database_name, table_name):
            if 'transacoes_cartao' in table_name:
                data = gerar_dados(table_name, num_records_update, clientes_id_usuarios)
                current_date = datetime.now().strftime("%d-%m-%Y")
                df = spark.createDataFrame(data, schema=StructType.fromJson(schema))
                df = df.withColumn(partition_by, lit(current_date))
            elif 'clientes' in table_name:
                data = gerar_dados(table_name, num_records_update)
                df = spark.createDataFrame(data, schema=StructType.fromJson(schema))
                # Apply bucketing for clientes table
                num_buckets = config.getint(table_name, 'num_buckets', fallback=5)
                df = df.repartition(num_buckets, "id_uf")
            else:
                data = gerar_dados(table_name, num_records_update)
                df = spark.createDataFrame(data, schema=StructType.fromJson(schema))
            
            df.createOrReplaceTempView("temp_view")
            logger.info("temp_view sample rows:")
            sample_rows = spark.sql(f"SELECT * FROM temp_view LIMIT 3").collect()
            for row in sample_rows:
                logger.info(str(row))

            update_table(spark, database_name, table_name, partition_by)
        else:
            logger.warning(f"Table '{table_name}' does not exist. Cannot update.")

    logger.info("Table update process completed")
    spark.stop()
    logger.debug("Spark session stopped")

if __name__ == "__main__":
    main()
