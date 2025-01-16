# -*- coding: utf-8 -*-
import sys, json, logging, os, argparse, configparser
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.functions import current_date

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.debug(f"Diretório de trabalho atual: {os.getcwd()}")
logger.debug(f"Conteúdo do diretório /app/mount/: {os.listdir('/app/mount/')}")

# Adicionar o diretório /app/mount/ ao sys.path
sys.path.append('/app/mount')

from utils import gerar_dados

def carregar_configuracao(config_path='/app/mount/config.ini'):
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
        
        config = configparser.ConfigParser()
        config.read(config_path)
        logger.info("Configuração carregada com sucesso.")
        logger.debug(f"Seções encontradas na configuração: {config.sections()}")
        return config
    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {str(e)}")
        raise

def tabela_existe(spark, nome_tabela):
    try:
        existe = spark.catalog.tableExists(nome_tabela)
        logger.info(f"Verificação de existência da tabela '{nome_tabela}': {'Existe' if existe else 'Não existe'}")
        return existe
    except Exception as e:
        logger.error(f"Erro ao verificar existência da tabela '{nome_tabela}': {str(e)}")
        raise

def criar_ou_atualizar_tabela(spark, nome_tabela, config):
    try:
        schema_base_path = '/app/mount/'
        schema_path = os.path.join(schema_base_path, f'{nome_tabela}.json')
        logger.info(f"Carregando esquema da tabela '{nome_tabela}' de: {schema_path}")
        
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Arquivo de esquema não encontrado: {schema_path}")

        with open(schema_path, 'r') as f:
            esquema = StructType.fromJson(json.load(f))
        logger.info(f"Esquema da tabela '{nome_tabela}' carregado com sucesso.")

        num_records = config.getint(nome_tabela, 'num_records')
        particionamento = config.getboolean(nome_tabela, 'particionamento')
        bucketing = config.getboolean(nome_tabela, 'bucketing')
        num_buckets = config.getint(nome_tabela, 'num_buckets')
        apenas_arquivos = config.getboolean('DEFAULT', 'apenas_arquivos', fallback=False)
        formato_arquivo = config['DEFAULT'].get('formato_arquivo', 'parquet')
        logger.info(f"Configurações para '{nome_tabela}':\n num_records={num_records},\n particionamento={particionamento},\n bucketing={bucketing},\n num_buckets={num_buckets}, \n only_files={apenas_arquivos},\n formato={formato_arquivo}")

        dados = gerar_dados(nome_tabela, num_records)
        df = spark.createDataFrame(dados, schema=esquema)
        df = df.withColumn("data_execucao", current_date())
        logger.info(f"Dados gerados para '{nome_tabela}': {num_records} registros")

        storage = config['storage'].get('storage_type', 'S3')
        if storage == 'S3':
            base_path = config['storage'].get('base_path')
        elif storage == 'ADLS':
            base_path = config['storage']['base_path']
        else:
            raise ValueError(f"Armazenamento não suportado: {storage}")

        if apenas_arquivos:
            output_path = f"{base_path}{nome_tabela}"
            write_options = {"mode": "overwrite", "format": formato_arquivo}
            
            if particionamento:
                df.write.partitionBy("data_execucao").options(**write_options).save(output_path)
                logger.info(f"Arquivos {formato_arquivo.upper()} para '{nome_tabela}' criados com particionamento por data_execucao em {output_path}")
            elif bucketing:
                df.write.bucketBy(num_buckets, "id_uf").options(**write_options).save(output_path)
                logger.info(f"Arquivos {formato_arquivo.upper()} para '{nome_tabela}' criados com bucketing por id_uf em {num_buckets} buckets em {output_path}")
            else:
                df.write.options(**write_options).save(output_path)
                logger.info(f"Arquivos {formato_arquivo.upper()} para '{nome_tabela}' criados sem particionamento ou bucketing em {output_path}")
        else:
            write_mode = "overwrite" if not tabela_existe(spark, nome_tabela) else "append"
            if write_mode == "append":
                spark.sql(f"REFRESH TABLE {nome_tabela}")
            write_options = {"mode": write_mode, "format": "parquet"}

            if particionamento:
                df.write.partitionBy("data_execucao").options(**write_options).saveAsTable(nome_tabela)
                logger.info(f"Tabela '{nome_tabela}' {write_mode} com particionamento por data_execucao")
            elif bucketing:
                df.write.bucketBy(num_buckets, "id_uf").options(**write_options).saveAsTable(nome_tabela)
                logger.info(f"Tabela '{nome_tabela}' {write_mode} com bucketing por id_uf em {num_buckets} buckets")
            else:
                df.write.options(**write_options).saveAsTable(nome_tabela)
                logger.info(f"Tabela '{nome_tabela}' {write_mode} sem particionamento ou bucketing")

    except Exception as e:
        logger.error(f"Erro ao criar ou atualizar tabela '{nome_tabela}': {str(e)}")
        raise

def main():
    try:
        config_path = '/app/mount/config.ini'
        config = carregar_configuracao(config_path)
        
        # Iniciar sessão Spark
        spark = SparkSession \
            .builder \
            .appName("SimulacaoDadosBancarios") \
            .enableHiveSupport() \
            .getOrCreate()
        logger.info("Sessão Spark iniciada com sucesso.")

        apenas_arquivos = config.getboolean('DEFAULT', 'apenas_arquivos', fallback=False)
        if not apenas_arquivos:
            database_name = config['DEFAULT'].get('dbname', 'bancodemo')
            spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")
            spark.sql(f"USE {database_name}")
            logger.info(f"Usando banco de dados: {database_name}")

        # Processamento das tabelas
        tabelas = config['DEFAULT'].get('tabelas', '').split(',')
        if tabelas:
            for tabela in tabelas:
                tabela = tabela.strip()
                logger.info(f"Processando tabela: '{tabela}'")
                if tabela in config.sections():
                    try:
                        criar_ou_atualizar_tabela(spark, tabela, config)
                    except Exception as e:
                        logger.error(f"Erro ao processar a tabela '{tabela}': {str(e)}")
                else:
                    logger.warning(f"Configuração não encontrada para a tabela '{tabela}'")
        else:
            logger.error("Nenhuma tabela especificada. Use o argumento --tabelas.")

    except Exception as e:
        logger.error(f"Erro na execução principal: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
