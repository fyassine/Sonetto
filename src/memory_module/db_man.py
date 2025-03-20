"""Manager database module.
Handles User Identification during Interactions."""
from pymongo import MongoClient
from copy import deepcopy

# Database setup
client = MongoClient("mongodb://localhost:27017/")
db = client["manager_db"]
customers = db["customers"]
embeddings = db["embeddings"]

