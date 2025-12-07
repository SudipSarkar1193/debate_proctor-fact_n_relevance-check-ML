# --- TOPIC REGISTRY ---
TOPIC_REGISTRY = {
    # EXISTING
    "ai": {
        "db_config": {
            "dbname": "debate_app_ai_db",
            "user": "postgres",
            "password": "root",
            "host": "localhost"
        },
        "categories": [
            "Category:Artificial_intelligence_controversies",
            "Category:Ethics_of_artificial_intelligence",
            "Category:Existential_risk_from_artificial_intelligence",
            "Category:Regulation_of_artificial_intelligence",
            "Category:Generative_artificial_intelligence"
        ]
    },

    # NEW
    "aadhaar": {
        "db_config": {
            "dbname": "debate_app_aadhaar_db",  
            "user": "postgres",
            "password": "root",
            "host": "localhost"
        },
        "categories": [
            # Main Identity Category (Contains Aadhaar articles)
            "Category:Identity documents of India", 
            
            # The Tech & Gov Initiatives
            "Category:Digital India initiatives",
            "Category:Biometrics",
            
            # Legal & Rights (Crucial for debate)
            "Category:Privacy in India",
            "Category:Human rights in India",
            "Category:Supreme_Court_of_India_cases",
            "Category:Censorship in India"
        ]
    }
}

# --- GLOBAL SETTINGS ---
# Scraper Settings 
MAX_DEPTH = 2  # How deep to go into subcategories (0 = top level only)
USER_AGENT = 'DebateAnalyzer_Bot/1.0 (contact: netajibosethesudip@gmail.com)'