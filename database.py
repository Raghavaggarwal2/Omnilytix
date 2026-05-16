import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from pymongo import MongoClient
from sqlalchemy.engine import URL


load_dotenv()


DEFAULT_SQL_SERVER_DRIVER = "ODBC Driver 18 for SQL Server"
DEFAULT_MONGODB_URI_ENV = "MONGODB_URI"
DEFAULT_MONGODB_DATABASE_ENV = "MONGODB_DATABASE"


@dataclass(frozen=True)
class DatabaseConfig:
    dialect: str
    host: str = ""
    port: int | None = None
    database: str = ""
    username: str = ""
    password: str = ""
    odbc_driver: str = DEFAULT_SQL_SERVER_DRIVER
    custom_uri: str | None = None


@dataclass(frozen=True)
class MongoDatabaseConfig:
    uri: str
    database: str


class MongoDatabaseConnection:
    def __init__(self, uri: str, database_name: str):
        self._client = MongoClient(uri)
        self.database_name = database_name

    @property
    def client(self) -> MongoClient:
        return self._client

    @property
    def database(self):
        return self._client[self.database_name]

    @property
    def dialect(self) -> str:
        return "mongodb"

    def list_collection_names(self) -> list[str]:
        return self.database.list_collection_names()

    def sample_documents(self, collection_name: str, limit: int = 5) -> list[dict[str, Any]]:
        cursor = self.database[collection_name].find({}).limit(limit)
        return list(cursor)

    def find_documents(
        self,
        collection_name: str,
        query: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        cursor = self.database[collection_name].find(query or {}).limit(limit)
        return list(cursor)


def get_database_uri_from_env() -> str:
    database_uri = os.getenv("DATABASE_URL")
    if not database_uri:
        raise RuntimeError(
            "Set DATABASE_URL to a SQLAlchemy URI, for example: "
            "postgresql+psycopg://user:password@localhost:5432/database_name"
        )
    return database_uri


def get_mongodb_uri_from_env() -> str:
    mongodb_uri = os.getenv(DEFAULT_MONGODB_URI_ENV)
    if not mongodb_uri:
        raise RuntimeError(
            "Set MONGODB_URI to a MongoDB connection string, for example: "
            "mongodb://user:password@localhost:27017"
        )
    return mongodb_uri


def get_mongodb_database_from_env() -> str:
    database_name = os.getenv(DEFAULT_MONGODB_DATABASE_ENV)
    if not database_name:
        raise RuntimeError("Set MONGODB_DATABASE to the database name you want to query.")
    return database_name


def build_database_uri(config: DatabaseConfig) -> str:
    if config.custom_uri:
        return config.custom_uri.strip()

    dialect = config.dialect.strip().lower()
    if not config.database:
        raise RuntimeError("Database name is required.")
    if not config.host:
        raise RuntimeError("Host is required.")

    username = config.username or None
    password = config.password or None
    port = config.port or None

    if dialect in {"postgres", "postgresql"}:
        url = URL.create(
            drivername="postgresql+psycopg",
            username=username,
            password=password,
            host=config.host,
            port=port,
            database=config.database,
        )
        return url.render_as_string(hide_password=False)

    if dialect in {"mysql", "mariadb"}:
        url = URL.create(
            drivername="mysql+pymysql",
            username=username,
            password=password,
            host=config.host,
            port=port,
            database=config.database,
        )
        return url.render_as_string(hide_password=False)

    if dialect in {"mssql", "sqlserver", "sql_server"}:
        query = {"driver": config.odbc_driver}
        if not password:
            query["trusted_connection"] = "yes"
        url = URL.create(
            drivername="mssql+pyodbc",
            username=None if not password else username,
            password=None if not password else password,
            host=config.host,
            port=port,
            database=config.database,
            query=query,
        )
        return url.render_as_string(hide_password=False)

    raise RuntimeError(
        "Unsupported database type. Choose PostgreSQL, MySQL, SQL Server, or provide a custom SQLAlchemy URI."
    )


def create_database_from_uri(database_uri: str) -> SQLDatabase:
    return SQLDatabase.from_uri(database_uri)


def create_database_from_config(config: DatabaseConfig) -> SQLDatabase:
    return create_database_from_uri(build_database_uri(config))


def create_database_from_env() -> SQLDatabase:
    return create_database_from_uri(get_database_uri_from_env())


def create_mongo_database_from_config(config: MongoDatabaseConfig) -> MongoDatabaseConnection:
    if not config.uri.strip():
        raise RuntimeError("MongoDB URI is required.")
    if not config.database.strip():
        raise RuntimeError("MongoDB database name is required.")
    return MongoDatabaseConnection(config.uri.strip(), config.database.strip())


def create_mongo_database_from_env() -> MongoDatabaseConnection:
    return create_mongo_database_from_config(
        MongoDatabaseConfig(
            uri=get_mongodb_uri_from_env(),
            database=get_mongodb_database_from_env(),
        )
    )