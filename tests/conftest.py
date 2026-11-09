"""Shared fixtures. DB tests run in a throwaway schema, never in the live table."""
import os
import uuid

import psycopg
import pytest


@pytest.fixture
def conn():
    from techpulse import store

    schema = f"test_{uuid.uuid4().hex}"
    with psycopg.connect(os.environ["TEST_DATABASE_URL"], autocommit=True) as connection:
        connection.execute(f'CREATE SCHEMA "{schema}"')
        connection.execute(f'SET search_path TO "{schema}"')
        store.init_schema(connection)
        yield connection
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
