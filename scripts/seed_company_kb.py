"""
Seed script: inserts ~100 realistic mid-size Indian companies into company_kb.sqlite.
Schema: cin (PK), name, state, city, status, category, paid_capital, lei_id
FTS indexes: name, city, state, category
"""
import sqlite3
import hashlib
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'company_kb.sqlite')

# Helper to generate a fake CIN (U + 5-digit NIC + 2-letter state + 4-digit year + company type + 6-digit num)
def make_cin(idx: int, state_code: str = "MH") -> str:
    return f"U{27100 + idx}{state_code}2005PTC{100000 + idx}"

# (name, city, state, category, paid_capital)
# category is used as keyword bag for FTS — include sector/product keywords
COMPANIES = [
    # --- JEWELLERY / GEMS (15) ---
    ("Rajkot Gems & Jewellery Pvt Ltd",        "Rajkot",       "GJ", "jewellery gems diamonds export manufacturing sme",    "5000000"),
    ("Shree Hari Diamond Exports Pvt Ltd",      "Surat",        "GJ", "jewellery diamonds polishing export manufacturing",   "8000000"),
    ("Jaipur Ratna Jewels Pvt Ltd",             "Jaipur",       "RJ", "jewellery gems precious stones kundan meenakari",    "3500000"),
    ("Tribhovandas Bhimji Zaveri Mid-Corp",     "Mumbai",       "MH", "jewellery gold silver retail manufacturing",         "12000000"),
    ("PC Chandra Gems Pvt Ltd",                 "Kolkata",      "WB", "jewellery gems gold silver retail",                  "6000000"),
    ("Gitanjali Jewels Wholesale Pvt Ltd",      "Mumbai",       "MH", "jewellery gold diamonds wholesale distribution",     "9000000"),
    ("Vaibhav Jewellers Pvt Ltd",               "Hyderabad",    "TG", "jewellery gold silver retail chains",                "4500000"),
    ("Shreenath Gems Exports Pvt Ltd",          "Rajkot",       "GJ", "jewellery gems export polishing diamonds",           "3000000"),
    ("Motisons Jewels Pvt Ltd",                 "Jaipur",       "RJ", "jewellery gold silver retail franchise",             "7000000"),
    ("Kiran Gems Pvt Ltd",                      "Surat",        "GJ", "gems diamonds polishing manufacturing export",       "15000000"),
    ("PNG Jewellers Mid-Corp Pvt Ltd",          "Pune",         "MH", "jewellery gold silver retail chains",                "5500000"),
    ("Senco Gold Wholesale Pvt Ltd",            "Kolkata",      "WB", "jewellery gold retail wholesale manufacturing",      "8500000"),
    ("Navratan Arts Jewels Pvt Ltd",            "Jaipur",       "RJ", "jewellery kundan meenakari gems artisan export",     "2500000"),
    ("Surat Diamond Bourse Traders Pvt Ltd",    "Surat",        "GJ", "diamonds gems polishing trading jewellery",          "11000000"),
    ("Rajmal Lakhichand Jewellers Pvt Ltd",     "Nagpur",       "MH", "jewellery gold silver retail regional chain",        "3200000"),

    # --- AUTO PARTS / COMPONENTS (20) ---
    ("Bharat Forge Mid Auto Pvt Ltd",           "Pune",         "MH", "auto parts forging components manufacturing OEM",   "20000000"),
    ("Sandhar Technologies Pune Pvt Ltd",       "Pune",         "MH", "auto parts locks mirrors die casting OEM",           "18000000"),
    ("Menon Pistons Pvt Ltd",                   "Kolhapur",     "MH", "auto parts pistons engine components manufacturing", "7000000"),
    ("Rane Engine Valve Pvt Ltd",               "Chennai",      "TN", "auto parts valves engine components OEM",            "9000000"),
    ("Minda Industries Component Pvt Ltd",      "Gurugram",     "HR", "auto parts switches horns lighting OEM",             "14000000"),
    ("Suprajit Engineering Components Pvt Ltd", "Bengaluru",    "KA", "auto parts cables controls manufacturing OEM",       "11000000"),
    ("Endurance Technologies Pune Pvt Ltd",     "Pune",         "MH", "auto parts die casting brakes suspension OEM",       "25000000"),
    ("Rico Auto Components Pvt Ltd",            "Gurugram",     "HR", "auto parts aluminium castings OEM manufacturing",    "16000000"),
    ("Automotive Axles Component Pvt Ltd",      "Mysuru",       "KA", "auto parts axles rear axle manufacturing OEM",       "8500000"),
    ("Lumax Auto Technologies Pvt Ltd",         "Gurugram",     "HR", "auto parts lighting systems OEM manufacturing",      "10000000"),
    ("Sharda Motor Components Pvt Ltd",         "Chennai",      "TN", "auto parts exhaust systems catalytic OEM",           "13000000"),
    ("Setco Automotive Clutch Pvt Ltd",         "Rajkot",       "GJ", "auto parts clutch assemblies manufacturing OEM",    "6000000"),
    ("JBM Auto Systems Pvt Ltd",                "Faridabad",    "HR", "auto parts sheet metal body components OEM",         "19000000"),
    ("Precision Camshafts Pvt Ltd",             "Solapur",      "MH", "auto parts camshafts engine components manufacturing","5500000"),
    ("Gabriel India Shocks Pvt Ltd",            "Pune",         "MH", "auto parts shock absorbers ride control OEM",        "12000000"),
    ("Ceekay Daikin Clutch Pvt Ltd",            "Pune",         "MH", "auto parts clutch friction OEM manufacturing",       "4500000"),
    ("Pricol Drive Solutions Pvt Ltd",          "Coimbatore",   "TN", "auto parts instrument clusters pumps OEM",           "9500000"),
    ("Neel Metal Products Pvt Ltd",             "Faridabad",    "HR", "auto parts stampings welded assemblies OEM",         "7500000"),
    ("Rolex Rings Auto Components Pvt Ltd",     "Rajkot",       "GJ", "auto parts bearing rings forging manufacturing",     "8000000"),
    ("SJS Enterprises Auto Pvt Ltd",            "Bengaluru",    "KA", "auto parts decorative components aesthetics OEM",    "4000000"),

    # --- PHARMA / API MANUFACTURERS (15) ---
    ("Divi's Laboratories Mid Pvt Ltd",         "Hyderabad",    "TG", "pharma API active pharmaceutical ingredients manufacturing export", "30000000"),
    ("Laurus Labs API Pvt Ltd",                 "Hyderabad",    "TG", "pharma API antiretroviral manufacturing export",     "22000000"),
    ("Granules India API Pvt Ltd",              "Hyderabad",    "TG", "pharma API paracetamol ibuprofen manufacturing",    "18000000"),
    ("Solara Active Pharma Pvt Ltd",            "Hyderabad",    "TG", "pharma API active ingredients contract manufacturing","15000000"),
    ("Sequent Scientific API Pvt Ltd",          "Hyderabad",    "TG", "pharma API veterinary human manufacturing export",   "12000000"),
    ("Dishman Carbogen Amcis Pvt Ltd",          "Ahmedabad",    "GJ", "pharma API CRAMS contract manufacturing export",    "20000000"),
    ("Cadila API Manufacturing Pvt Ltd",        "Ahmedabad",    "GJ", "pharma API formulations manufacturing generic",      "25000000"),
    ("Zydus Wellness Pharma Pvt Ltd",           "Ahmedabad",    "GJ", "pharma OTC formulations consumer healthcare",       "14000000"),
    ("Alembic Pharma API Pvt Ltd",              "Vadodara",     "GJ", "pharma API antibiotics manufacturing export",       "16000000"),
    ("Smruthi Organics API Pvt Ltd",            "Solapur",      "MH", "pharma API specialty chemicals manufacturing",      "6000000"),
    ("Neuland Laboratories Pvt Ltd",            "Hyderabad",    "TG", "pharma API complex molecules manufacturing export", "10000000"),
    ("Aurbindo Pharma Component Pvt Ltd",       "Hyderabad",    "TG", "pharma API penicillin antibiotic manufacturing",    "28000000"),
    ("Concord Biotech API Pvt Ltd",             "Ahmedabad",    "GJ", "pharma API fermentation biocatalysis manufacturing","8000000"),
    ("Suven Pharmaceuticals Pvt Ltd",           "Hyderabad",    "TG", "pharma API CNS CRAMS contract manufacturing",       "9000000"),
    ("Kopran API Chemicals Pvt Ltd",            "Mumbai",       "MH", "pharma API bulk drugs manufacturing export",        "7000000"),

    # --- TEXTILES (15) ---
    ("Surat Textile Mills Pvt Ltd",             "Surat",        "GJ", "textiles synthetic fabrics weaving polyester export","10000000"),
    ("Vardhman Fabrics Mid Pvt Ltd",            "Ludhiana",     "PB", "textiles yarn cotton spinning manufacturing",       "15000000"),
    ("Tiruppur Knitwear Exports Pvt Ltd",       "Tiruppur",     "TN", "textiles knitwear hosiery garments export",        "8000000"),
    ("Arvind Textiles Component Pvt Ltd",       "Ahmedabad",    "GJ", "textiles denim fabrics weaving manufacturing",      "20000000"),
    ("Welspun India Fabrics Pvt Ltd",           "Ahmedabad",    "GJ", "textiles home textiles terry towels export",        "18000000"),
    ("Nahar Industrial Textiles Pvt Ltd",       "Ludhiana",     "PB", "textiles yarn spinning cotton polyester",           "12000000"),
    ("Bombay Rayon Fabrics Pvt Ltd",            "Mumbai",       "MH", "textiles fabrics woven shirting suiting",          "9000000"),
    ("Kitex Garments Pvt Ltd",                  "Kizhakkambalam","KL","textiles garments babywear export manufacturing",   "7000000"),
    ("Alok Industries Fabrics Pvt Ltd",         "Surat",        "GJ", "textiles polyester fabric weaving integrated",      "11000000"),
    ("Gokaldas Exports Garments Pvt Ltd",       "Bengaluru",    "KA", "textiles garments apparel export manufacturing",    "13000000"),
    ("Trident Textiles Pvt Ltd",                "Ludhiana",     "PB", "textiles yarn terry towels home export",           "16000000"),
    ("Pasupati Fabrics Pvt Ltd",                "Delhi",        "DL", "textiles polyester fabrics weaving manufacturing",  "5000000"),
    ("Indian Terrain Garments Pvt Ltd",         "Chennai",      "TN", "textiles menswear garments retail brands",          "6500000"),
    ("K.P.R. Mill Textiles Pvt Ltd",            "Coimbatore",   "TN", "textiles yarn spinning knitwear garments integrated","14000000"),
    ("Raymond Fabrics Mid Pvt Ltd",             "Thane",        "MH", "textiles suiting shirting worsted wool fabrics",    "17000000"),

    # --- IT SERVICES / SAAS MID-SIZE (15) ---
    ("Mphasis IT Services Pvt Ltd",             "Bengaluru",    "KA", "IT services software BPO digital cloud consulting", "22000000"),
    ("Mastech Holdings Tech Pvt Ltd",           "Bengaluru",    "KA", "IT services staffing digital transformation",       "9000000"),
    ("Newgen Software Pvt Ltd",                 "Delhi",        "DL", "IT SaaS digital process automation platform",       "8000000"),
    ("Tata Elxsi Embedded Pvt Ltd",             "Bengaluru",    "KA", "IT engineering services embedded R&D design",       "18000000"),
    ("Birlasoft IT Services Pvt Ltd",           "Noida",        "UP", "IT services ERP SAP digital consulting",            "15000000"),
    ("Happiest Minds Technologies Pvt Ltd",     "Bengaluru",    "KA", "IT services digital transformation cloud security", "10000000"),
    ("Zensar Technologies Pvt Ltd",             "Pune",         "MH", "IT services digital experience engineering cloud",  "14000000"),
    ("Persistent Systems Pvt Ltd",              "Pune",         "MH", "IT software engineering digital services cloud",    "20000000"),
    ("KPIT Technologies Pvt Ltd",               "Pune",         "MH", "IT automotive software engineering embedded",       "12000000"),
    ("Intellect Design Arena Pvt Ltd",          "Chennai",      "TN", "IT SaaS fintech banking platform digital",          "11000000"),
    ("Majesco Software Pvt Ltd",                "Mumbai",       "MH", "IT SaaS insurance platform cloud digital",          "7000000"),
    ("Subex Analytics Pvt Ltd",                 "Bengaluru",    "KA", "IT SaaS telecom analytics fraud management",        "6000000"),
    ("Nucleus Software Exports Pvt Ltd",        "Noida",        "UP", "IT SaaS lending banking platform fintech",          "8500000"),
    ("Rategain Travel Tech Pvt Ltd",            "Noida",        "UP", "IT SaaS travel hospitality revenue management",     "9500000"),
    ("Kellton Tech Solutions Pvt Ltd",          "Hyderabad",    "TG", "IT services digital blockchain IoT consulting",     "5000000"),

    # --- STEEL / METALS FABRICATION (10) ---
    ("Pennar Industries Steel Pvt Ltd",         "Hyderabad",    "TG", "steel fabrication structural sections manufacturing","12000000"),
    ("Man Industries Steel Pvt Ltd",            "Anjar",        "GJ", "steel pipes tubes manufacturing energy sector",     "10000000"),
    ("Shyam Metalics Fabrication Pvt Ltd",      "Howrah",       "WB", "steel metals sponge iron billets fabrication",     "18000000"),
    ("Welspun Corp Steel Pvt Ltd",              "Mumbai",       "MH", "steel pipes line pipe manufacturing energy",        "20000000"),
    ("Jindal Stainless Components Pvt Ltd",     "Jamshedpur",   "JH", "steel stainless fabrication coils sheets",         "15000000"),
    ("Uttam Value Steels Pvt Ltd",              "Wardha",       "MH", "steel hot rolled coils sheets manufacturing",      "9000000"),
    ("Kamdhenu Structural Steel Pvt Ltd",       "Bhiwadi",      "RJ", "steel TMT bars structural manufacturing",          "7000000"),
    ("Maharashtra Seamless Tubes Pvt Ltd",      "Nagpur",       "MH", "steel seamless pipes tubes oil gas manufacturing",  "13000000"),
    ("Shankara Building Products Pvt Ltd",      "Bengaluru",    "KA", "steel metals building products distribution",       "8000000"),
    ("Gallantt Metal Steels Pvt Ltd",           "Gorakhpur",    "UP", "steel TMT bars sponge iron manufacturing",         "6000000"),

    # --- AGRICULTURE / FOOD PROCESSING (10) ---
    ("Choudhary Agro Food Pvt Ltd",             "Ludhiana",     "PB", "agriculture food processing rice basmati export",  "4000000"),
    ("LT Foods Processing Pvt Ltd",             "Gurugram",     "HR", "agriculture food processing basmati rice export",  "9000000"),
    ("Patanjali Foods Processing Pvt Ltd",      "Haridwar",     "UK", "agriculture food processing FMCG edible oil",      "15000000"),
    ("Prabhat Dairy Foods Pvt Ltd",             "Ahmednagar",   "MH", "agriculture dairy food processing milk products",  "7000000"),
    ("Kohinoor Speciality Foods Pvt Ltd",       "Mumbai",       "MH", "agriculture food processing rice speciality brands","6000000"),
    ("Avanti Feeds Processing Pvt Ltd",         "Hyderabad",    "TG", "agriculture aquaculture shrimp feed processing",   "11000000"),
    ("Flex Foods Pvt Ltd",                      "Dehradun",     "UK", "agriculture food processing dehydrated vegetables", "3500000"),
    ("Mahindra Agri Solutions Pvt Ltd",         "Mumbai",       "MH", "agriculture agri-inputs seeds crop protection",    "8000000"),
    ("McCain Foods India Pvt Ltd",              "Mehsana",      "GJ", "agriculture food processing frozen potato products","10000000"),
    ("Kwality Dairy Foods Pvt Ltd",             "Delhi",        "DL", "agriculture dairy milk products processing",       "5000000"),
]

assert len(COMPANIES) == 100, f"Expected 100 companies, got {len(COMPANIES)}"


def seed(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    inserted = 0
    for idx, (name, city, state, category, paid_capital) in enumerate(COMPANIES):
        cin = make_cin(idx, state)
        try:
            conn.execute(
                """INSERT OR IGNORE INTO company_kb
                   (cin, name, state, city, status, category, paid_capital, lei_id)
                   VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?, '')""",
                (cin, name, state, city, category, paid_capital),
            )
            if conn.execute("SELECT changes()").fetchone()[0] > 0:
                inserted += 1
        except sqlite3.Error as e:
            print(f"  ERROR inserting {name!r}: {e}")

    conn.commit()

    # Rebuild FTS index
    conn.execute("INSERT INTO company_fts(company_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    return inserted


def test_search(db_path: str):
    conn = sqlite3.connect(db_path)
    tests = [
        ("jewellery",  "jewellery"),
        ("auto parts", "auto parts"),
        ("pharma",     "pharma"),
        ("textiles",   "textiles"),
        ("IT SaaS",    "IT SaaS"),
        ("steel",      "steel"),
        ("agriculture","agriculture"),
    ]
    all_pass = True
    for query, label in tests:
        # FTS5 multi-word needs quotes or AND
        fts_query = f'"{query}"' if " " in query else query
        rows = conn.execute(
            "SELECT name, city FROM company_fts WHERE company_fts MATCH ? LIMIT 5",
            (fts_query,)
        ).fetchall()
        status = "PASS" if len(rows) >= 2 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] '{label}' -> {len(rows)} results: {[r[0] for r in rows[:3]]}")
    conn.close()
    return all_pass


if __name__ == "__main__":
    db_path = os.path.normpath(DB_PATH)
    print(f"DB path: {db_path}")
    print(f"Seeding {len(COMPANIES)} companies...")
    n = seed(db_path)
    print(f"Inserted {n} new rows (INSERT OR IGNORE skips existing).")
    print()
    print("FTS search tests:")
    ok = test_search(db_path)
    print()
    if ok:
        print("All search tests PASSED.")
    else:
        print("Some search tests FAILED — check categories above.")
