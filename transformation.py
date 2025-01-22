import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_spark_session(app_name="Transformation"):
    """
    Initialize a Spark session.

    Args:
        app_name (str): The name of the Spark application.

    Returns:
        SparkSession: The initialized Spark session.
    """
    logger.info("Initializing Spark session")
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_tables(spark):
    """
    Load the clientes and transacoes_cartao tables.

    Args:
        spark (SparkSession): The active Spark session.

    Returns:
        tuple: A tuple containing the clientes and transacoes_cartao DataFrames.
    """
    logger.info("Loading tables")
    clientes = spark.sql("SELECT id_usuario FROM clientes")
    transacoes_cartao = spark.sql("SELECT * FROM transacoes_cartao")
    logger.debug(f"Loaded {clientes.count()} records from clientes")
    logger.debug(f"Loaded {transacoes_cartao.count()} records from transacoes_cartao")
    return clientes, transacoes_cartao

def repeat_clientes(clientes, transacoes_count):
    """
    Repeat and randomize the id_usuario column from clientes if necessary.

    Args:
        clientes (DataFrame): The clientes DataFrame.
        transacoes_count (int): The number of records in transacoes_cartao.

    Returns:
        DataFrame: The repeated and randomized clientes DataFrame.
    """
    clientes_count = clientes.count()
    logger.debug(f"Clientes count: {clientes_count}, Transacoes count: {transacoes_count}")
    if clientes_count < transacoes_count:
        logger.info("Repeating and randomizing id_usuario from clientes")
        repetition_factor = (transacoes_count // clientes_count) + 1
        clientes_repeated = clientes.withColumn("repeat", rand()).repartition("repeat")
        clientes_repeated = clientes_repeated.selectExpr("id_usuario").rdd.flatMap(lambda x: [x] * repetition_factor).toDF(["id_usuario"])
    else:
        logger.info("No need to repeat clientes")
        clientes_repeated = clientes
    return clientes_repeated

def update_transacoes_cartao(spark, clientes_repeated):
    """
    Update the transacoes_cartao table with id_usuario from clientes_repeated.

    Args:
        spark (SparkSession): The active Spark session.
        clientes_repeated (DataFrame): The repeated and randomized clientes DataFrame.

    Returns:
        DataFrame: The updated transacoes_cartao DataFrame.
    """
    logger.info("Updating transacoes_cartao with id_usuario from clientes_repeated")
    clientes_repeated.createOrReplaceTempView("clientes_repeated")
    updated_transacoes = spark.sql("""
        SELECT t.*, c.id_usuario
        FROM transacoes_cartao t
        JOIN clientes_repeated c
        ON RAND() < 1.0 / (SELECT COUNT(*) FROM clientes_repeated)
    """)
    return updated_transacoes

def save_updated_transacoes(updated_transacoes):
    """
    Overwrite the existing transacoes_cartao table with the updated data.

    Args:
        updated_transacoes (DataFrame): The updated transacoes_cartao DataFrame.
    """
    logger.info("Saving updated transacoes_cartao table")
    updated_transacoes.write.mode("overwrite").saveAsTable("transacoes_cartao")
    logger.info("Updated transacoes_cartao table saved successfully")

def main():
    """
    Main function to execute the transformation script.
    """
    spark = initialize_spark_session()
    clientes, transacoes_cartao = load_tables(spark)
    clientes_repeated = repeat_clientes(clientes, transacoes_cartao.count())
    updated_transacoes = update_transacoes_cartao(spark, clientes_repeated)
    save_updated_transacoes(updated_transacoes)
    logger.info("Transformation completed successfully")

if __name__ == "__main__":
    main()
