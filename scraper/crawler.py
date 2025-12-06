import wikipediaapi
from config import USER_AGENT, MAX_DEPTH

# Initialize the API wrapper
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent=USER_AGENT,
    language='en'
)

def extract_page_data(page_obj):
    """Converts a Wiki Page Object into a clean dictionary."""
    return {
        "pageid": page_obj.pageid,
        "title": page_obj.title,
        "url": page_obj.fullurl,
        "content": page_obj.text, # Takes full text
        "categories": list(page_obj.categories.keys())
    }

def fetch_category_pages(category_name, level=0):
    """
    Generator that yields page data from a category.
    Handles recursion based on MAX_DEPTH.
    """
    cat_page = wiki_wiki.page(category_name)
    
    if not cat_page.exists():
        print(f"   ⚠️ Category '{category_name}' not found.")
        return

    print(f"📂 Scanning Category: {category_name} (Level {level})")

    for member in cat_page.categorymembers.values():
        
        # Case 1: It's an Article
        if member.ns == wikipediaapi.Namespace.MAIN:
            yield extract_page_data(member)
        
        # Case 2: It's a Subcategory (Recursion)
        elif member.ns == wikipediaapi.Namespace.CATEGORY and level < MAX_DEPTH:
            yield from fetch_category_pages(member.title, level + 1)