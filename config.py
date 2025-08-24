import os

GROQ_API_KEY = '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'
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  
LLM_MODEL = "llama3-8b-8192"  # Using a reliable Groq model 
CHUNK_SIZE = 512
CHUNK_OVERLAP = 60
VECTOR_DB_DIR = "vector_db"

# Environment settings to prevent PyTorch issues
os.environ["USE_TF"] = "0"  # disable tensorflow backend
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF logs if any
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent tokenizer warnings
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # fallback for M1/M2 Macs

# SSL settings for corporate networks
import ssl
import urllib3

# Disable SSL warnings for corporate firewalls
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# SSL bypass options (uncomment if in corporate environment with SSL issues)
SSL_VERIFY = False  # Set to False to bypass SSL verification
os.environ["CURL_CA_BUNDLE"] = ""  # Disable SSL verification for requests
os.environ["REQUESTS_CA_BUNDLE"] = ""  # Disable SSL verification for requests
