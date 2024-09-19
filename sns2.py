import requests
import heapq
import numpy as np
import pandas as pd  # To handle CSV data
from sklearn.neural_network import MLPRegressor
import math

# Constants for Open-Meteo Marine Weather API
BASE_URL = "https://marine-api.open-meteo.com/v1/marine"

# Haversine formula to calculate curved distance between two points on Earth
def haversine_distance(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Radius of Earth in kilometers
    R = 6371.0
    
    # Distance in kilometers
    distance = R * c
    return distance

# Fetch weather data from the Open-Meteo API for a specific coordinate (latitude, longitude)
def fetch_weather_data(lat, lon):
    url = (f"{BASE_URL}?latitude={lat}&longitude={lon}&current=wave_height,wave_direction,"
           "wind_wave_height,wind_wave_direction,ocean_current_velocity,ocean_current_direction"
           "&hourly=wave_height,wave_direction,wind_wave_height,wind_wave_direction"
           "&daily=wave_height_max,wave_direction_dominant,wind_wave_height_max,wind_wave_direction_dominant"
           "&timezone=auto")
    response = requests.get(url)
    return response.json()

# Convert weather data to numerical values, create a weather grid with cost for A* algorithm
def fetch_and_convert_weather_data(grid):
    weather_grid = {}
    for lat, lon in grid:
        weather_data = fetch_weather_data(lat, lon)
        
        # Safely extract wave height and wind speed, using 0 if they are missing
        wave_height = weather_data.get('hourly', {}).get('wave_height', [None])[0]
        if wave_height is None:
            wave_height = 0  # Default to 0 if not available
        
        wind_speed = weather_data.get('current', {}).get('ocean_current_velocity', 0)
        if wind_speed is None:
            wind_speed = 0  # Default to 0 if not available
        
        # Calculate cost based on wave height and wind speed
        cost = 1 + wind_speed * 0.1 + wave_height * 0.2
        
        # Store numerical weather data in the grid
        weather_grid[(lat, lon)] = {
            'wind_speed': wind_speed,
            'wave_height': wave_height,
            'cost': cost  # This is used for A* cost function
        }
    
    return weather_grid

# Create a grid of geographic points
def create_grid(lat_min, lat_max, lon_min, lon_max, resolution):
    grid = []
    for lat in range(int(lat_min * 10), int(lat_max * 10), int(resolution * 10)):
        for lon in range(int(lon_min * 10), int(lon_max * 10), int(resolution * 10)):
            grid.append((lat / 10.0, lon / 10.0))  # Add each grid cell as a tuple
    return grid

# Artificial Neural Network (ANN) for fuel and RPM prediction (mock data)
def train_ann(X, y):
    ann_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500)
    ann_model.fit(X, y)
    return ann_model

# A* Algorithm for route optimization using the weather grid data
def a_star_search(start, goal, weather_grid):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: haversine_distance(start, goal)}
    
    while open_list:
        _, current = heapq.heappop(open_list)
        
        if current == goal:
            return reconstruct_path(came_from, current)
        
        for neighbor in get_neighbors(current):
            if neighbor in weather_grid and is_navigable(neighbor, weather_grid):
                temp_g_score = g_score[current] + weather_grid[neighbor]['cost']
                
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = g_score[neighbor] + haversine_distance(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
    return None

# Reconstruct the optimal path found by A* algorithm
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

# Check if a grid cell is navigable based on weather conditions
def is_navigable(lat_lon, weather_grid):
    wind_speed = weather_grid[lat_lon]['wind_speed']
    wave_height = weather_grid[lat_lon]['wave_height']
    # Define navigability based on thresholds for wind and wave
    if wind_speed > 20 or wave_height > 6:  # Example thresholds
        return False
    return True

# Get neighbors for a given grid point (4-connectivity: up, down, left, right)
def get_neighbors(current):
    lat, lon = current
    return [(lat + 0.5, lon), (lat - 0.5, lon), (lat, lon + 0.5), (lat, lon - 0.5)]

# Load ports data from CSV and convert to DataFrame
def load_ports_data(file_path):
    ports_df = pd.read_csv(file_path)
    return ports_df

# Choose two ports as start and goal points
def select_ports_for_test(ports_df):
    start_port = ports_df.sample().iloc[0]
    goal_port = ports_df.sample().iloc[0]
    start = (start_port['Latitude'], start_port['Longitude'])
    goal = (goal_port['Latitude'], goal_port['Longitude'])
    
    print(f"Start Port: {start_port['Port Name']} ({start})")
    print(f"Goal Port: {goal_port['Port Name']} ({goal})")
    
    return start, goal

# Main function
def main():
    # Load the ports dataset
    ports_df = load_ports_data('ports.csv')  # Ensure the file path is correct
    
    # Select start and goal ports from the dataset
    start, goal = select_ports_for_test(ports_df)
    
    # Define geographic area and resolution for the grid
    lat_min, lat_max = min(start[0], goal[0]) - 1, max(start[0], goal[0]) + 1
    lon_min, lon_max = min(start[1], goal[1]) - 1, max(start[1], goal[1]) + 1
    resolution = 0.5  # Grid resolution in degrees
    
    # Create grid of points
    grid = create_grid(lat_min, lat_max, lon_min, lon_max, resolution)
    
    # Fetch weather data and convert to numerical values (cost)
    weather_grid = fetch_and_convert_weather_data(grid)
    
    # Simulate ANN training data and train the ANN model
    X_train, y_train = simulate_ann_training_data()
    ann_model = train_ann(X_train, y_train)
    
    # Run the A* algorithm to find the optimal path between selected ports
    optimal_route = a_star_search(start, goal, weather_grid)
    
    if optimal_route:
        print("Optimal Route:", optimal_route)
    else:
        print("No navigable route found.")
    
if __name__ == "__main__":
    main()
