import wikipediaapi

# Based on the Wikipedia Category page you found
CATEGORIES_TO_SCAN = [
    # The Main Technical Definitions
    "Category:Artificial_intelligence",
    "Category:Generative_artificial_intelligence",
    "Category:Machine_learning",

    # The "Debate" Topics (Critical for your project)
    "Category:Artificial_intelligence_controversies", # <--- Index 3
    "Category:Criticism_of_artificial_intelligence",
    "Category:Ethics_of_artificial_intelligence",
    "Category:Existential_risk_from_artificial_intelligence",
    "Category:Regulation_of_artificial_intelligence",
    "Category:Philosophy_of_artificial_intelligence",
    "Category:Deaths_caused_by_robots_and_artificial_intelligence", 
    "Category:AI_safety"
]

# 1. SETUP
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='MyAI_Test_Script/1.0 (contact: test@example.com)',
    language='en'
)

# Global counter
count = 0
MAX_LIMIT = 100
MAX_LEVEL = 2
captured_titles = []

def print_categorymembers(categorymembers, level=0):
    global count
    for c in categorymembers.values():
        if count >= MAX_LIMIT or level >= MAX_LEVEL:
            return

        indent = "  " * level
        
        if c.ns == wikipediaapi.Namespace.MAIN:
            print(f"{indent}[ARTICLE] {c.title}")
            captured_titles.append(c.title)
            count += 1
        elif c.ns == wikipediaapi.Namespace.CATEGORY:
            print(f"{indent}[CATEGORY] {c.title}")
            print_categorymembers(c.categorymembers, level + 1)

def test_single_page_content(page_title):
    print(f"\n\n{'='*40}")
    print(f"TESTING PAGE CONTENT: {page_title}")
    print(f"{'='*40}")
    
    page = wiki_wiki.page(page_title)
    if page.exists():
        print(f"URL: {page.fullurl}")
        print("-" * 20)
        print(page.text[:1000] + "\n\n... [Content Truncated] ...") 
    else:
        print(f"Page '{page_title}' does not exist.")

# --- EXECUTION ---


target_cat_name = CATEGORIES_TO_SCAN[3] # (Artificial_intelligence_controversies)

print(f"🚀 Fetching first {MAX_LIMIT} pages from: {target_cat_name}...\n")

cat = wiki_wiki.page(target_cat_name)

if cat.exists():
    print_categorymembers(cat.categorymembers)
    
    # Test the first article found
    if captured_titles:
        test_single_page_content(captured_titles[0])
    else:
        print("\n❌ No articles found in this category.")
else:
    print(f"\n❌ Error: The category '{target_cat_name}' does not exist.")