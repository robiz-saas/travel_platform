import os, requests
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from meteostat import Point, Monthly
from pydantic import BaseModel
from langchain.tools import StructuredTool
import dateparser

import re

def get_weather_or_climate(city: str, date_str: str) -> str:
    """
    city: Name of the city (e.g., "Bangalore")
    date_str: Natural language date (e.g., "tomorrow", "next week", "2025-06-20", "second week of October")
    """
    try:
        # Try parsing a specific date first
        parsed_date = dateparser.parse(date_str, settings={'PREFER_DATES_FROM': 'future'})

        # Handle "week of" or "second week of October" expressions
        week_match = re.search(r"(\bfirst|\bsecond|\bthird|\bfourth|\blast)?\s*week\s+of\s+([a-zA-Z]+\s*\d{0,4})", date_str.lower())
        if week_match:
            week_word, month_expr = week_match.groups()
            week_num = {
                "first": 1, "second": 2, "third": 3, "fourth": 4, "last": -1
            }.get(week_word or "first", 1)

            base_date = dateparser.parse(month_expr.strip(), settings={'PREFER_DATES_FROM': 'future'})
            if not base_date:
                return f"Could not parse month from '{month_expr}'."

            # Get first day of that month
            first_of_month = datetime(base_date.year, base_date.month, 1)
            # Calculate all Mondays (week starts) in that month
            mondays = [first_of_month + timedelta(days=i) for i in range(31)
                       if (first_of_month + timedelta(days=i)).month == base_date.month
                       and (first_of_month + timedelta(days=i)).weekday() == 0]
            if not mondays:
                return f"Could not determine weeks in {month_expr}."

            # Pick the correct week
            if week_num == -1:
                start_of_week = mondays[-1]
            elif week_num <= len(mondays):
                start_of_week = mondays[week_num - 1]
            else:
                return f"{week_word.title()} week not found in {month_expr}."

            end_of_week = start_of_week + timedelta(days=6)

            parsed_date_range = (start_of_week.date(), end_of_week.date())
        else:
            if not parsed_date:
                return f"Could not understand the date '{date_str}'. Try something like 'tomorrow' or 'next Friday'."
            parsed_date = parsed_date.date()
            parsed_date_range = (parsed_date, parsed_date)

        today = datetime.today().date()

        # Geocoding
        geolocator = Nominatim(user_agent="travel-assistant")
        location = geolocator.geocode(city)
        if not location:
            return f"Could not find location for '{city}'."

        latitude, longitude = location.latitude, location.longitude

        start_date, end_date = parsed_date_range

        # --- Short-term (Forecast) ---
        if end_date <= today + timedelta(days=5):
            api_key = os.getenv("OPENWEATHER_API_KEY")
            if not api_key:
                return "OpenWeather API key missing."

            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
            response = requests.get(url)
            data = response.json()

            if data.get("cod") != "200":
                return f"Error: {data.get('message', 'Failed to get forecast')}"

            # Filter forecasts for the date range
            forecasts = [
                f for f in data["list"]
                if start_date <= datetime.strptime(f["dt_txt"], "%Y-%m-%d %H:%M:%S").date() <= end_date
            ]

            if not forecasts:
                return f"No forecast available for {city} between {start_date} and {end_date}."

            detailed_forecast = []
            for f in forecasts:
                detailed_forecast.append(
                    f"""
                    Time: {f['dt_txt']}
                    - Weather: {f['weather'][0]['description'].capitalize()}
                    - Temperature: {f['main']['temp']}°C (Feels like: {f['main']['feels_like']}°C)
                    - Min/Max: {f['main']['temp_min']}°C / {f['main']['temp_max']}°C
                    - Humidity: {f['main']['humidity']}%
                    - Pressure: {f['main']['pressure']} hPa
                    - Wind: {f['wind']['speed']} m/s, Direction: {f['wind'].get('deg', 'N/A')}°
                    - Cloud Cover: {f['clouds']['all']}%
                    - Visibility: {f.get('visibility', 0) / 1000} km
                    """.strip()
                )

            return f"📅 Detailed weather forecast for {city.title()} from {start_date} to {end_date}:\n\n" + "\n\n".join(detailed_forecast)

        # --- Long-term (Climate) ---
        else:
            point = Point(latitude, longitude)
            end = datetime.today()
            start = end.replace(year=end.year - 10)
            data = Monthly(point, start, end).fetch()

            if data.empty:
                return f"No historical climate data found for {city}."

            avg_stats = {"tavg": [], "tmin": [], "tmax": [], "prcp": [], "snow": []}
            for single_date in [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]:
                month = single_date.month
                monthly_data = data[data.index.month == month]
                if monthly_data.empty:
                    continue
                avg_stats["tavg"].append(monthly_data["tavg"].mean())
                avg_stats["tmin"].append(monthly_data["tmin"].mean())
                avg_stats["tmax"].append(monthly_data["tmax"].mean())
                avg_stats["prcp"].append(monthly_data["prcp"].mean())
                if "snow" in monthly_data:
                    avg_stats["snow"].append(monthly_data["snow"].mean())

            if not avg_stats["tavg"]:
                return f"No climate data found for {city} in the given range."

            return (
                f"📊 Typical climate in {city.title()} from {start_date.strftime('%b %d')} to {end_date.strftime('%b %d')} (10-year average):\n"
                f"- Avg. Temperature: {sum(avg_stats['tavg'])/len(avg_stats['tavg']):.1f}°C\n"
                f"- Min/Max Temperature: {sum(avg_stats['tmin'])/len(avg_stats['tmin']):.1f}°C / {sum(avg_stats['tmax'])/len(avg_stats['tmax']):.1f}°C\n"
                f"- Precipitation: {sum(avg_stats['prcp'])/len(avg_stats['prcp']):.1f} mm\n"
                f"- Snowfall: {sum(avg_stats['snow'])/len(avg_stats['snow']) if avg_stats['snow'] else 0:.1f} mm\n"
            )

    except Exception as e:
        return f"Error: {str(e)}"

# Tool schema for LangChain agent
class WeatherInput(BaseModel):
    city: str
    date_str: str

weather_forecast_tool = StructuredTool.from_function(
    name="Weather_forecast",
    description="Get weather or climate info for a city and date. Accepts natural dates like 'tomorrow' or 'next week'.",
    func=get_weather_or_climate,
    args_schema=WeatherInput,
)