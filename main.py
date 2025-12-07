import argparse
from database import operations as db
from scraper import crawler
from config import TOPIC_REGISTRY

def main():
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Ingest Wikipedia data for a specific topic.")
    parser.add_argument("--topic", type=str, required=True, help="The topic key (e.g., 'ai', 'aadhaar')")
    args = parser.parse_args()
    
    topic = args.topic.lower()

    # Validate Topic
    if topic not in TOPIC_REGISTRY:
        print(f"❌ Error: Topic '{topic}' not found in config.py")
        print(f"   Available topics: {list(TOPIC_REGISTRY.keys())}")
        return

    print(f"🚀 Starting Phase 1: Ingestion Engine for TOPIC: [{topic.upper()}]")
    
    # Load Topic Config
    target_categories = TOPIC_REGISTRY[topic]["categories"]

    # Ensure DB is ready (for this specific topic)
    db.init_db(topic)

    total_saved = 0

    # Loop through categories for this topic
    for category in target_categories:
        print(f"\n--- Processing: {category} ---")
        
        # Crawler remains generic (it just fetches what we tell it)
        pages_generator = crawler.fetch_category_pages(category)
        
        for page_data in pages_generator:
            
            # Check DB specific to this topic
            if db.page_exists(topic, page_data['pageid']):
                print(f"   ⏩ Exists: {page_data['title']}")
                continue 

            if not page_data['content']:
                continue

            # Save to the specific topic DB
            success = db.save_page(topic, page_data)
            
            if success:
                print(f"   💾 Saved: {page_data['title']}")
                total_saved += 1
            else:
                print(f"   ⏩ Skipped: {page_data['title']}")

    print(f"\n🎉 Ingestion Complete for {topic.upper()}!")
    print(f"Total pages stored in {TOPIC_REGISTRY[topic]['db_config']['dbname']}: {total_saved}")

if __name__ == "__main__":
    main()