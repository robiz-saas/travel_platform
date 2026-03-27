import random
import string
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv
from groq import Groq
import json

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_user_id():
    return "U" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

 

def llm_suggest_items(destination, purpose, current_items):
    prompt = f"""
You're a smart travel assistant helping users pack for trips.

Task: Suggest exactly 5 *additional* one-word packing items for a user visiting {destination} for {purpose}.

Already packed: {', '.join(current_items)}

Respond ONLY with a valid JSON list of 5 lowercase one-word items like:
["passport", "charger", "umbrella", "sneakers", "glasses"]
"""


    response = client.chat.completions.create(
        model="llama3-8b-8192",  # or try "mixtral-8x7b-32768"
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        print("Failed to parse LLM output. Raw response:")
        print(response.choices[0].message.content)
        return []
