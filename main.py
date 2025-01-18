from dotenv import load_dotenv

load_dotenv()

import openai
import requests
import json
from typing import Optional, Dict, List, Any
import os
from geopy.geocoders import Nominatim

# Define the weather tool schema
WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get weather forecast for a US city by first getting coordinates and then fetching the forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                    "state": {
                        "type": "string",
                        "description": "The two-letter state code",
                    },
                },
                "required": ["city", "state"],
            },
        },
    }
]


def get_coordinates(city: str, state: str) -> Optional[Dict[str, float]]:
    """Get latitude and longitude for a US city using Geopy"""
    geolocator = Nominatim(user_agent="WeatherAssistant/1.0")
    location = geolocator.geocode(f"{city}, {state}, USA")

    if location:
        return {"latitude": location.latitude, "longitude": location.longitude}
    else:
        return None


def get_forecast(latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
    """Get weather forecast from NWS API using coordinates"""

    try:
        # First get the grid points
        point_url = f"https://api.weather.gov/points/{latitude},{longitude}"
        point_response = requests.get(point_url)
        point_response.raise_for_status()
        point_data = point_response.json()

        # Get the forecast URL from the points response
        forecast_url = point_data["properties"]["forecast"]
        forecast_response = requests.get(forecast_url)
        forecast_response.raise_for_status()

        return forecast_response.json()
    except Exception as e:
        print(f"Error getting forecast: {e}")
        return None


def get_weather_forecast(city: str, state: str) -> str:
    """Function that will be called by the OpenAI tool"""
    coords = get_coordinates(city, state)
    if not coords:
        return f"Could not find coordinates for {city}, {state}"

    forecast = get_forecast(coords["latitude"], coords["longitude"])
    if not forecast:
        return f"Could not get forecast for {city}, {state}"

    # Extract the next few periods of forecast
    periods = forecast["properties"]["periods"][:3]

    # Format the response
    response = f"Weather forecast for {city}, {state}:\n\n"
    for period in periods:
        response += f"{period['name']}:\n"
        response += (
            f"Temperature: {period['temperature']}°{period['temperatureUnit']}\n"
        )
        response += f"Conditions: {period['shortForecast']}\n"
        response += f"Wind: {period['windSpeed']} {period['windDirection']}\n\n"

    return response


def process_weather_query(query: str) -> str:
    """Process a natural language weather query using OpenAI's tools"""

    try:
        # Make the initial request to OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # or your preferred model
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": query},
            ],
            tools=WEATHER_TOOLS,
        )

        # Check if there are tool calls in the response
        if response.choices[0].message.tool_calls:
            # Determine which tool is being called
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)

            # Call the weather function with the extracted parameters
            result = get_weather_forecast(
                city=function_args["city"], state=function_args["state"]
            )
        else:
            # Display any other responses
            result = response.choices[0].message.content

        return result

    except Exception as e:
        return f"Error processing weather query: {e}"


# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Example queries
    example_queries = [
        "Say hello",
        "I might be interested in the weather, would you know how to get a weather report?",
        "What's the weather like in Seattle, Washington?",
    ]

    for query in example_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        result = process_weather_query(query)
        print(result)
        print("=" * 50)
