from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["smart_travel_assistant"]

travel_details = db["travel_details"]
checklists = db["checklists"]

def save_travel_details(user_data):
    travel_details.insert_one(user_data)

def save_checklist(user_id, items):
    checklists.update_one(
        {"userId": user_id},
        {"$set": {"items": items}},
        upsert=True
    )

def get_checklist(user_id):
    return checklists.find_one({"userId": user_id})

def get_travel_info(user_id):
    return travel_details.find_one({"userId": user_id})
