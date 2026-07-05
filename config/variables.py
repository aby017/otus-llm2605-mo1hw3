import os
from dotenv import load_dotenv

load_dotenv()

YC_API_KEY = os.getenv("YC_API_KEY")
YC_CATALOG_ID = os.getenv("YC_CATALOG_ID")