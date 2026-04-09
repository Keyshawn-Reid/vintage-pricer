import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

app_id = os.getenv("EBAY_APP_ID")

url = "https://svcs.ebay.com/services/search/FindingService/v1"

params = {
    "OPERATION-NAME": "findItemsAdvanced",
    "SERVICE-VERSION": "1.0.0",
    "SECURITY-APPNAME": app_id,
    "RESPONSE-DATA-FORMAT": "JSON",
    "keywords": "harley davidson vintage tee",
    "itemFilter(0).name": "ListingType",
    "itemFilter(0).value": "AuctionWithBIN",
    "paginationInput.entriesPerPage": "20"
}

response = requests.get(url, params=params)
print("Status code:", response.status_code)
print(response.json())