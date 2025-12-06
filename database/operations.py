# database/operations.py
import psycopg2
from config import DB_CONFIG

def get_connection():
    """Establishes connection to Postgres."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Database Connection Error: {e}")
        raise e

def init_db():
    """Creates the necessary table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS raw_facts (
        pageid INTEGER PRIMARY KEY,
        title TEXT UNIQUE,
        url TEXT,
        content TEXT,
        categories TEXT,
        source TEXT DEFAULT 'wikipedia',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
        conn.commit()
        print("✅ Database initialized (Table 'raw_facts' ready).")
    except Exception as e:
        print(f"❌ Error creating table: {e}")
    finally:
        conn.close()

def page_exists(pageid):
    """Checks if a page ID already exists in the DB."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM raw_facts WHERE pageid = %s", (pageid,))
            return cur.fetchone() is not None
    finally:
        conn.close()

def save_page(page_data):
    """
    Saves a single page dictionary to the DB.
    Ignores duplicates using ON CONFLICT.
    """
    sql = """
    INSERT INTO raw_facts (pageid, title, url, content, categories)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (pageid) DO NOTHING;
    """
    
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (
                page_data['pageid'],
                page_data['title'],
                page_data['url'],
                page_data['content'],
                str(page_data['categories']) # Store list as string representation
            ))
        conn.commit()
        return True
    except Exception as e:
        print(f"⚠️ DB Error saving '{page_data['title']}': {e}")
        conn.rollback()
        return False
    finally:
        conn.close()