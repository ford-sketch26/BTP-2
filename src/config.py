from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

load_dotenv(PROJECT_ROOT / ".env")

CTGOV_API_BASE = "https://clinicaltrials.gov/api/v2"
DEFAULT_PAGE_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 30

# Auto-discovery (see data_fetcher.discover_sponsor_variants) finds CT.gov
# sponsor names that share a token with the parent — e.g. "Gilead" matches
# "Kite, A Gilead Company". It misses acquisitions whose CT.gov sponsor
# string contains no token from the parent name — e.g. Janssen for JNJ,
# Hospira for PFE. SPONSOR_OVERRIDES lists those known M&A relationships
# explicitly. Extend as you encounter new cases.
SPONSOR_OVERRIDES: dict[str, list[str]] = {
    "PFE":  ["Pfizer", "Wyeth", "Pharmacia", "Hospira", "Array BioPharma", "Medivation"],
    "ABBV": ["AbbVie", "Allergan", "Pharmacyclics"],
    "JNJ":  ["Johnson & Johnson", "Janssen", "Actelion"],
    "BMY":  ["Bristol-Myers Squibb", "Celgene", "Juno Therapeutics"],
    "LLY":  ["Eli Lilly", "Loxo Oncology", "Dermira"],
    "MRK":  ["Merck Sharp & Dohme", "Merck", "Schering-Plough"],
    "AZN":  ["AstraZeneca", "MedImmune", "Alexion Pharmaceuticals"],
    "NVS":  ["Novartis", "Sandoz", "Alcon", "AveXis"],
    "GSK":  ["GlaxoSmithKline", "GSK", "ViiV Healthcare", "Tesaro"],
    "RHHBY": ["Hoffmann-La Roche", "Genentech", "Roche"],
    "SNY":  ["Sanofi", "Genzyme", "Bioverativ", "Ablynx"],
    "AMGN": ["Amgen", "Onyx Pharmaceuticals", "Five Prime Therapeutics"],
    "VRTX": ["Vertex Pharmaceuticals"],
    "REGN": ["Regeneron Pharmaceuticals"],
    "BIIB": ["Biogen"],
}
