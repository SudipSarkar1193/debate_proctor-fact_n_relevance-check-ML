# database/operations.py
import psycopg2
from config import TOPIC_REGISTRY

def get_connection(topic):
    """
    Establishes connection to the SPECIFIC database for the given topic.
    """
    if topic not in TOPIC_REGISTRY:
        raise ValueError(f"❌ Unknown topic: {topic}")

    # Load specific DB config for this topic
    db_config = TOPIC_REGISTRY[topic]["db_config"]
    
    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except Exception as e:
        print(f"❌ Database Connection Error ({topic}): {e}")
        raise e

def init_db(topic):
    """Creates the table in the specific topic's database."""
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
    conn = get_connection(topic)
    try:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
        conn.commit()
        print(f"✅ Database initialized for '{topic}' (Table 'raw_facts' ready).")
    except Exception as e:
        print(f"❌ Error creating table for '{topic}': {e}")
    finally:
        conn.close()

def page_exists(topic, pageid):
    """Checks if a page ID already exists in the specific topic DB."""
    conn = get_connection(topic)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM raw_facts WHERE pageid = %s", (pageid,))
            return cur.fetchone() is not None
    finally:
        conn.close()

def save_page(topic, page_data):
    """
    Saves a page to the specific topic's database.
    """
    sql = """
    INSERT INTO raw_facts (pageid, title, url, content, categories)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (pageid) DO NOTHING;
    """
    
    conn = get_connection(topic)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (
                page_data['pageid'],
                page_data['title'],
                page_data['url'],
                page_data['content'],
                str(page_data['categories'])
            ))
        conn.commit()
        return True
    except Exception as e:
        print(f"⚠️ DB Error saving '{page_data['title']}' to '{topic}': {e}")
        conn.rollback()
        return False
    finally:
        conn.close()