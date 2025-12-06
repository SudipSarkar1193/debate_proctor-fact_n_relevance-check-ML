# Database Credentials
DB_CONFIG = {
    "dbname": "debate_app_ai_db",
    "user": "postgres",
    "password": "root", 
    "host": "localhost"
}

# The Target List
TARGET_CATEGORIES = [
    "Category:Artificial_intelligence_controversies",
    "Category:Ethics_of_artificial_intelligence",
    "Category:Existential_risk_from_artificial_intelligence",
    "Category:Regulation_of_artificial_intelligence",
    "Category:Generative_artificial_intelligence"
]

# Scraper Settings
MAX_DEPTH = 2  # How deep to go into subcategories (0 = top level only)
USER_AGENT = 'DebateAnalyzer_Bot/1.0 (contact: netajibosethesudip@gmail.com)'