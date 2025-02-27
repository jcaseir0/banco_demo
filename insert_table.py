import logging, sys
from configparser import ConfigParser
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import StructType, StringType, IntegerType, DoubleType, TimestampType
from pyspark.sql.functions import lit
from common_functions import load_config, gerar_dados, table_exists
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_spark_session(jdbc_url: str, thrift_server: str) -> SparkSession:
    """Creates and configures a Spark session.

    Args:
        jdbc_url (str): JDBC URL for Hive metastore.
        thrift_server (str): Thrift server URI.

    Returns:
        SparkSession: Configured Spark session.
    """
    logger.info("Creating Spark session")
    try:
        spark_conf = SparkConf()
        spark_conf.set("hive.metastore.client.factory.class", "com.cloudera.spark.hive.metastore.HivemetastoreClientFactory")
        spark_conf.set("hive.metastore.uris", thrift_server)
        spark_conf.set("spark.sql.hive.metastore.jars", "builtin")
        spark_conf.set("spark.sql.hive.hiveserver2.jdbc.url", jdbc_url)

        spark = SparkSession.builder.config(conf=spark_conf).appName("UpdateTable").enableHiveSupport().getOrCreate()
        logger.info("Spark session created successfully")
        return spark
    except Exception as e:
        logger.error(f"Error creating Spark session: {e}")
        raise

def get_table_columns(spark: SparkSession, database_name: str, table_name: str) -> list:
    """
    Retrieves a list of valid column names from the table schema.

    This function fetches the schema of the specified table and extracts a list of column
    names, excluding partition information and special columns (e.g., '# col_name', 'data_type').

    Args:
        spark (SparkSession): The active Spark session.
        database_name (str): The name of the database containing the table.
        table_name (str): The name of the table.

    Returns:
        list: A list of valid column names for the table.

    Raises:
        Exception: If an error occurs while retrieving the table schema.
    """
    logger.debug(f"Retrieving table schema for {database_name}.{table_name}")
    try:
        df = spark.table(f"{database_name}.{table_name}")
        columns = df.columns
        logger.info(f"Columns: {', '.join(columns)}")
        return columns
    except Exception as e:
        logger.error(f"Error retrieving table schema for {database_name}.{table_name}: {str(e)}")
        raise

def insert_data(spark: SparkSession, database_name: str, table_name: str, columns: list,
                partition_by: str = None, is_bucketed: bool = False) -> None:
    """
    Inserts data into the specified table, handling partitioning and bucketing.

    This function inserts data into the specified table from a temporary view.
    It supports tables that are partitioned, bucketed, or neither.

    Args:
        spark (SparkSession): The active Spark session.
        database_name (str): The name of the database containing the table.
        table_name (str): The name of the table to update.
        columns (list): A list of valid column names for the table.
        partition_by (str, optional): The column to partition by, if any. Defaults to None.
        is_bucketed (bool, optional): Whether the table is bucketed. Defaults to False.

    Raises:
        Exception: If an error occurs during the data insertion process.
    """
    logger.info(f"Inserting data into table: {database_name}.{table_name}")
    logger.debug(f"Partition by: {partition_by}, Is bucketed: {is_bucketed}")

    try:
        column_list = ", ".join(columns)
        logger.debug(f"Inserting Columns: {column_list}")

        if partition_by:
            current_date = datetime.now().strftime("%d-%m-%Y")
            logger.debug(f"Inserting data with partition: {partition_by}='{current_date}'")
            spark.sql(f"""
                INSERT INTO {database_name}.{table_name}
                PARTITION ({partition_by}='{current_date}')
                SELECT {column_list}
                FROM temp_view
            """)
        elif is_bucketed:
            logger.debug("Inserting data into bucketed table with bucketing column")
            spark.sql(f"""
                INSERT INTO {database_name}.{table_name}
                SELECT {column_list}
                FROM temp_view
            """)
        else:
            logger.debug("Inserting data without partition or bucketing")
            spark.sql(f"""
                INSERT INTO {database_name}.{table_name} 
                SELECT * 
                FROM temp_view
            """)

        logger.info(f"Data inserted into table '{table_name}' successfully.")
    except Exception as e:
        logger.error(f"Error inserting data into table '{table_name}': {str(e)}")
        raise

def generate_and_write_data(spark: SparkSession, config: ConfigParser, table_name: str, clientes_data=None) -> None:
    """Generates data and writes it to the specified table.

    Args:
        spark (SparkSession): The active Spark session.
        config (ConfigParser): The configuration object.
        table_name (str): The name of the table to update.
        clientes_data (list): Data for the 'clientes' table, structured as a list of dictionaries.
    """
    database_name = config.get("DEFAULT", "dbname")
    logger.debug(f"Database name: {database_name}")
    logger.info(f"Generating and writing data for table: {database_name}.{table_name}")
    
    try:
        num_records_update = config.getint(table_name, 'num_records_update', fallback=100)
        logger.debug(f"Number of records to update: {num_records_update}")
        partition_by = config.get(table_name, 'partition_by', fallback=None)
        bucketing_column = config.get(table_name, 'clustered_by', fallback=None)
        is_bucketed = config.getboolean(table_name, 'bucketing', fallback=False)
        logger.debug(f"Is bucketed: {is_bucketed}")
        num_buckets = config.getint(table_name, 'num_buckets', fallback=5) if is_bucketed else 0
        logger.debug(f"Partition by: {partition_by}, Bucketing column: {bucketing_column}, Num Buckets: {num_buckets}")

        if 'transacoes_cartao' in table_name:
            clientes_ids = [cliente['id_usuario'] for cliente in clientes_data] if clientes_data else None
            data = gerar_dados(table_name, num_records_update, clientes_ids)
        elif 'clientes' in table_name:
            data = gerar_dados(table_name, num_records_update) if clientes_data is None else clientes_data
        else:
            data = gerar_dados(table_name, num_records_update)

        record_count_before = spark.sql(f"SELECT COUNT(*) FROM {database_name}.{table_name}").collect()[0][0]
        logger.info(f"Total records in table '{table_name}' before insert: {record_count_before}")

        columns = get_table_columns(spark, database_name, table_name)
        logger.info(f"Columns: {columns}")

        table_schema = spark.table(f"{database_name}.{table_name}").schema
        df = spark.createDataFrame(data, schema=table_schema)

        if is_bucketed:
            logger.info(f"Repartitioning dataframe by {bucketing_column} into {num_buckets} buckets")
            df = df.repartition(num_buckets, bucketing_column)

        df.createOrReplaceTempView("temp_view")

        insert_data(spark, database_name, table_name, columns, partition_by, is_bucketed)
        
        record_count_after = spark.sql(f"SELECT COUNT(*) FROM {database_name}.{table_name}").collect()[0][0]
        logger.info(f"Total records in table '{table_name}' after insert: {record_count_after}")

        sample_rows = spark.sql(f"SELECT * FROM {database_name}.{table_name} LIMIT 3").collect()
        logger.info(f"Sample rows from table '{table_name}':")
        for row in sample_rows:
            logger.info(str(row))

        if 'transacoes_cartao' in table_name:
            transacoes_sample = spark.sql(f"SELECT id_usuario FROM {database_name}.{table_name} LIMIT 3").collect()
            transacoes_ids = [row.id_usuario for row in transacoes_sample]
            logger.info(f"Sampled id_usuario from 'transacoes_cartao' table: {transacoes_ids}")

            clientes_sample = spark.sql(f"""
                SELECT id_usuario
                FROM {database_name}.clientes
                WHERE id_usuario IN ('{"','".join(transacoes_ids)}')
                LIMIT 3
            """).collect()

            clientes_ids = [row.id_usuario for row in clientes_sample]
            logger.info(f"Sampled id_usuario from 'clientes' table: {clientes_ids}")

    except Exception as e:
        logger.error(f"Error generating and writing data for table '{table_name}': {e}", exc_info=True)
        raise

def generate_clientes_data(config: ConfigParser) -> list:
    """
    Generates client data based on the configuration.
    """
    logger.info("Generating clientes data")
    try:
        clientes_table = [table for table in config.get("DEFAULT", "tables").split(",") if 'clientes' in table][0]
        logger.debug(f"Clientes table: {clientes_table}")
        clientes_num_records = config.getint(clientes_table, 'num_records_update', fallback=100)
        clientes_data = gerar_dados(clientes_table, clientes_num_records)
        logger.debug(f"Clientes data generated sample: {clientes_data[:3]}")
        return clientes_data
    
    except Exception as e:
        logger.error(f"Error generating clientes data: {e}")
        return []

def main():
    """
    Main function to update tables based on configuration.

    This function loads the configuration, iterates through the tables,
    generates new data, and updates each existing table.
    """
    logger.info("Starting table update process")
    config = load_config()
    logger.debug("Configuration loaded")

    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return
    logger.debug("Configuration loaded")

    spark = None
    try:
        jdbc_url = sys.argv[1]
        logger.debug(f"JDBC URL: {jdbc_url}")

        server_dns = jdbc_url.split('//')[1].split('/')[0]
        thrift_server = f"thrift://{server_dns}:9083"
        logger.debug(f"Thrift Server: {thrift_server}")

        spark = create_spark_session(jdbc_url, thrift_server)
        database_name = config.get("DEFAULT", "dbname")
        tables = config.get("DEFAULT", "tables").split(",")
        logger.debug(f"Tables: {tables}")

        clientes_data = generate_clientes_data(config)
        logger.debug(f"Clientes table: {clientes_data}")
                     
        for table_name in tables:
            table_name = table_name.strip()
            logger.info(f"Processing table: {table_name}")

            if table_exists(spark, database_name, table_name):
                try:
                    generate_and_write_data(spark, config, table_name, clientes_data)
                except Exception as e:
                    logger.error(f"Failed to generate and write data for table '{table_name}': {e}")
            else:
                logger.warning(f"Table '{table_name}' does not exist. Cannot update.")
        logger.info("Table update process completed")
    except Exception as e:
        logger.error(f"Error updating tables: {e}", exc_info=True)
    finally:
        if spark:
            logger.info("Stopping Spark session")
            spark.stop()
            logger.debug("Spark session stopped")

if __name__ == "__main__":
    main()