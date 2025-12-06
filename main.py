from database import operations as db
from scraper import crawler
from config import TARGET_CATEGORIES

def main():
    print("🚀 Starting Phase 1: Ingestion Engine...")
    
    # 1. Ensure DB is ready
    db.init_db()

    total_saved = 0

    # 2. Loop through our config list
    for category in TARGET_CATEGORIES:
        print(f"\n--- Processing: {category} ---")
        
        # Get the generator from the scraper
        pages_generator = crawler.fetch_category_pages(category)
        
        for page_data in pages_generator:

            # OPTIMIZATION: Check DB before processing
            if db.page_exists(page_data['pageid']):
                print(f"   ⏩ Exists: {page_data['title']}")
                continue  # Skip to next page immediately
            
            # Simple validation: Skip empty pages
            if not page_data['content']:
                continue

            # 3. Save to DB
            success = db.save_page(page_data)
            
            if success:
                print(f"   💾 Saved: {page_data['title']}")
                total_saved += 1
            else:
                print(f"   ⏩ Skipped (Duplicate or Error): {page_data['title']}")

    print(f"\n🎉 Ingestion Complete!")
    print(f"Total pages stored in Postgres: {total_saved}")

if __name__ == "__main__":
    main()