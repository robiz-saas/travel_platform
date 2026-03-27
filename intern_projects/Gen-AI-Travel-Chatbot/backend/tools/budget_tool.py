import json, re
from pydantic import BaseModel
from langchain.tools import tool
from langchain_core.output_parsers import PydanticOutputParser

@tool
def estimate_budget_tool(destination: str, duration: int, style: str, num_people: int = 1, age_group: str = "") -> str:
    """
    Estimate budget components based on destination, duration, travel style, number of travelers, and optional age group.
    Returns JSON with hotel, food, activities, and transport cost.
    Always show all prices in Indian Rupees (₹), even if the destination is international
    """
    city_base_rates = {
        "goa": {"hotel": 1000, "food": 400, "activities": 300},
        "manali": {"hotel": 900, "food": 350, "activities": 250},
        "dubai": {"hotel": 3500, "food": 1200, "activities": 1000},
        "paris": {"hotel": 5000, "food": 2000, "activities": 1200},
    }
    style_multipliers = {"budget": 1.0, "midrange": 1.8, "luxury": 3.5}
    age_modifiers = {
        "kids": {"food": 0.7, "activities": 0.6, "hotel": 1.0},
        "elderly": {"food": 1.0, "activities": 0.8, "hotel": 1.2},
    }

    rates = city_base_rates.get(destination.lower(), city_base_rates["goa"])
    multiplier = style_multipliers.get(style.lower(), 1.8)
    age_mod = age_modifiers.get(age_group.lower(), {"food": 1.0, "activities": 1.0, "hotel": 1.0})

    hotel_price_per_night = int(rates["hotel"] * multiplier * age_mod["hotel"])
    food_cost_per_day = int(rates["food"] * multiplier * age_mod["food"])
    activity_cost_per_day = int(rates["activities"] * multiplier * age_mod["activities"])

    return json.dumps({
        "hotel_price_per_night": hotel_price_per_night,
        "food_cost_per_day": food_cost_per_day,
        "activity_cost_per_day": activity_cost_per_day,
        "transport_cost": 2000,
        "num_people": num_people
    })

class TravelModel(BaseModel):
    destination: str
    duration: int
    style: str
    num_people: int = 1
    age_group: str = ""  # optional: "kids", "elderly", or empty
    hotels: list[str] = []
    restaurants: list[str] = []
    activities: list[str] = []
    places_to_visit: list[str] = []

def format_travel_summary(data: TravelModel, tool_output: dict) -> str:
    num_people = data.num_people or 1

    hotel_total = tool_output["hotel_price_per_night"] * data.duration * num_people
    food_total = tool_output["food_cost_per_day"] * data.duration * num_people
    activity_total = tool_output["activity_cost_per_day"] * len(data.activities) * num_people
    total = hotel_total + food_total + activity_total + tool_output["transport_cost"]

    summary = f"""
{data.duration}-Day {data.style.title()} Trip to {data.destination.title()} for {num_people} {"people" if num_people > 1 else "person"}{f" with {data.age_group}" if data.age_group else ""}:

Hotels:
""" + ''.join(f"- {h} (₹{tool_output['hotel_price_per_night']:,}/night/person)\n" for h in data.hotels)

    summary += "\nRestaurants:\n" + ''.join(f"- {r}\n" for r in data.restaurants)
    summary += "\nPlaces:\n" + ''.join(f"- {p}\n" for p in data.places_to_visit)
    summary += "\nActivities:\n" + ''.join(f"- {a}\n" for a in data.activities)

    summary += f"""
\n*Total Est. Cost*: ₹{total:,}
  • Hotel: ₹{hotel_total:,}
  • Food: ₹{food_total:,}
  • Activities: ₹{activity_total:,}
  • Transport: ₹{tool_output['transport_cost']:,}
"""
    return summary.strip()