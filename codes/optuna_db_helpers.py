# =============================================================================
# optuna_db_helpers.py
# -----------------------------------------------------------------------------
# Project  : Deconstructing Oversampling in Software Defect Prediction:
#            Algorithm Constraints, Trade-offs, and New Baselines
# Purpose  : Utility helpers for managing the PostgreSQL backend used by
#            Optuna.  Provides fast study deletion and database recreation
#            routines to keep the optimisation storage clean between
#            experimental runs.
# =============================================================================

import time

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

class OptunaDBHelpers:
    @staticmethod
    def fast_delete_study(study_name):
        db_config = {
            "dbname": "optuna_db",
            "user": "optuna",
            "password": "optuna",
            "host": "localhost",
            "port": 5432
        }

        try:
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            cursor.execute("SELECT study_id FROM studies WHERE study_name = %s;", (study_name,))
            study_id = cursor.fetchone()

            if study_id is None:
                return

            study_id = study_id[0]

            cursor.execute("DELETE FROM trial_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = %s);", (study_id,))
            cursor.execute("DELETE FROM trial_params WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = %s);", (study_id,))
            cursor.execute("DELETE FROM trial_user_attributes WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = %s);", (study_id,))
            cursor.execute("DELETE FROM trial_system_attributes WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = %s);", (study_id,))
            cursor.execute("DELETE FROM trials WHERE study_id = %s;", (study_id,))
            cursor.execute("DELETE FROM study_directions WHERE study_id = %s;", (study_id,))
            cursor.execute("DELETE FROM study_user_attributes WHERE study_id = %s;", (study_id,))
            cursor.execute("DELETE FROM study_system_attributes WHERE study_id = %s;", (study_id,))
            cursor.execute("DELETE FROM studies WHERE study_id = %s;", (study_id,))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            print(f"Error: {e}")


    @staticmethod
    def fast_recreate():
        db_config = {
            "dbname": "postgres",  # Connect to the default postgres DB to run drop/create
            "user": "optuna",
            "password": "optuna",
            "host": "localhost",
            "port": 5432
        }
        try:
            conn = psycopg2.connect(**db_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = 'optuna_db'
                AND pid <> pg_backend_pid();
            """)

            time.sleep(0.5)

            cursor.execute("""
                SELECT count(*) 
                FROM pg_stat_activity 
                WHERE datname = 'optuna_db';
            """)
            remaining = cursor.fetchone()[0]

            if remaining > 0:
                time.sleep(1)

            cursor.execute("DROP DATABASE IF EXISTS optuna_db;")
            cursor.execute("CREATE DATABASE optuna_db WITH OWNER = optuna;")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def create_indexes():
        db_config = {
            "dbname": "optuna_db",
            "user": "optuna",
            "password": "optuna",
            "host": "localhost",
            "port": 5432
        }

        try:
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_study_id ON trials (study_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trial_id ON trial_params (trial_id);")

            conn.commit()
            cursor.close()
            conn.close()
            print("Indexes created successfully!")

        except Exception as e:
            print(f"Error: {e}")