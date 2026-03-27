import yagmail
import os
from dotenv import load_dotenv

load_dotenv()

SENDER_EMAIL = os.getenv("GMAIL_USER")
APP_PASSWORD = os.getenv("GMAIL_PASS")

yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)

def send_welcome_email(email, name, destination, start_date, end_date, purpose, companions, checklist, user_id):
    items = "\n - " + "\n - ".join([item["item"] for item in checklist]) if checklist else "\n(Empty)"

    body = f"""
Hi {name},

Welcome to Smart Travel Assistant!

Your trip details:
- Destination: {destination}
- Dates: {start_date} to {end_date}
- Purpose: {purpose}
- Companions: {', '.join(companions)}

Your checklist:{items}

✅ Your User ID: {user_id}

Keep this ID to view or update your details later.

Safe travels!  
– Smart Travel Assistant
"""
    yag.send(to=email, subject="🎉 Trip Created: Smart Travel Assistant", contents=body)

def send_checklist_update_email(email, checklist_items, name):
    items = "\n - " + "\n - ".join([item["item"] for item in checklist_items])
    body = f"""
Hi {name},

Your packing checklist was just updated!

Current items:{items}

Don't forget to mark priority items.

Cheers,
Smart Travel Assistant
"""
    yag.send(to=email, subject="🧳 Your Checklist Was Updated", contents=body)

def send_reminder_email(email, checklist_items, name, destination, start_date):
    all_items = ""
    for item in checklist_items:
        mark = "⭐" if item.get("priority") else ""
        all_items += f"\n - {item['item']} {mark}"

    body = f"""
Hi {name},

Just a friendly reminder: your trip to {destination} starts on {start_date} — only 2 days left!

Here's your packing checklist:
{all_items}

Items marked with ⭐ are your priority items.

Wishing you a smooth and well-packed journey! ✈️  
- Smart Travel Assistant
"""
    yag.send(to=email, subject=f"📅 Reminder: Your {destination} trip is coming up!", contents=body)
