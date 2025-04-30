# preReq installations: pip install dash plotly pandas numpy

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import math
import typing  # Import the typing module


class Location:
    """Represents a geographical location with latitude, longitude, and altitude."""

    def __init__(self, latitude: float, longitude: float, altitude: float = 0.0):
        if not (-90 <= latitude <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees.")
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude  # Assuming meters or feet, specify if needed

    def __repr__(self):
        return f"Location(lat={self.latitude:.6f}, lon={self.longitude:.6f}, alt={self.altitude})"

    def __eq__(self, other):
        if isinstance(other, Location):
            return (abs(self.latitude - other.latitude) < 1e-6 and
                    abs(self.longitude - other.longitude) < 1e-6 and
                    abs(self.altitude - other.altitude) < 1e-6)
        return False


class Drone:
    """Represents a single drone with its state."""

    def __init__(self, drone_id: str, velocity: float, battery_lvl: int, current_location: Location):
        self.drone_id = drone_id
        # e.g., in meters per second (float for precision)
        self.velocity = velocity
        self.battery_lvl = battery_lvl  # e.g., percentage 0-100
        self.current_location = current_location
        self._current_command_index = 0  # Keep track of the current command being executed

        if not (0 <= self.battery_lvl <= 100):
            print(f"Warning: Battery level {self.battery_lvl} for {
                  self.drone_id} is outside expected range [0, 100].")

    def __repr__(self):
        return (f"Drone(id='{self.drone_id}', velocity={self.velocity}, "
                f"battery={self.battery_lvl}%, location={self.current_location})")

    def get_current_command(self, commands: typing.List['Command']) -> typing.Optional['Command']:
        if 0 <= self._current_command_index < len(commands):
            # Ensure the command belongs to this drone
            if commands[self._current_command_index].drone_id == self.drone_id:
                return commands[self._current_command_index]
            else:
                print(f"Warning: Command at index {self._current_command_index} is for drone {
                      commands[self._current_command_index].drone_id}, not {self.drone_id}. Skipping.")
                pass  # Continue without returning a command if it's not for this drone

        return None

    def advance_command(self):
        self._current_command_index += 1


class Waypoint:
    """Represents a navigation waypoint."""

    def __init__(self, waypoint_id: str, name: str, location: Location):
        self.waypoint_id = waypoint_id
        self.name = name
        self.location = location

    def __repr__(self):
        return f"Waypoint(id='{self.waypoint_id}', name='{self.name}', location={self.location})"


class Command:
    """Base class for all drone commands."""

    def __init__(self, command_id: str, drone_id: str):
        self.command_id = command_id
        self.drone_id = drone_id  # The ID of the drone this command is for

    def __repr__(self):
        return f"Command(id='{self.command_id}', drone_id='{self.drone_id}')"


class MovementCommand(Command):
    """Represents a command for drone movement."""

    def __init__(self, command_id: str, drone_id: str, direction: int, turn: str, goto_waypoint_id: str):
        super().__init__(command_id, drone_id)
        # e.g., heading in degrees (can be simplified for this demo)
        self.direction = direction
        # e.g., "left", "right", "none" (can be ignored for this demo)
        self.turn = turn
        self.goto_waypoint_id = goto_waypoint_id  # Reference to a Waypoint by ID

    def __repr__(self):
        return (f"MovementCommand(id='{self.command_id}', drone_id='{self.drone_id}', "
                f"direction={self.direction}, turn='{self.turn}', goto_waypoint_id='{self.goto_waypoint_id}')")


class InternalCommand(Command):
    """Represents a command for internal drone actions."""

    def __init__(self, command_id: str, drone_id: str, is_charging: bool = False, is_calibrating: bool = False, is_processing: bool = False):
        super().__init__(command_id, drone_id)
        self.is_charging = is_charging
        self.is_calibrating = is_calibrating
        self.is_processing = is_processing

        # Ensure only one internal action is true at a time (optional, depends on logic)
        # active_actions = sum([self.is_charging, self.is_calibrating, self.is_processing])
        # if active_actions > 1:
        #     print(f"Warning: Multiple internal actions true for command {self.command_id}.")

    def __repr__(self):
        actions = []
        if self.is_charging:
            actions.append("charging")
        if self.is_calibrating:
            actions.append("calibrating")
        if self.is_processing:
            actions.append("processing")
        action_str = ", ".join(actions) if actions else "none"
        return (f"InternalCommand(id='{self.command_id}', drone_id='{self.drone_id}', "
                f"actions=[{action_str}])")


class Mission:
    """Represents a mission plan containing drones, waypoints, and commands."""

    def __init__(self, name: str, start_time: int, end_time: int):
        self.name = name
        self.start_time = start_time  # Unix timestamp or similar integer representation
        self.end_time = end_time   # Unix timestamp or similar integer representation
        # Store drones by drone_id
        self.drones: typing.Dict[str, Drone] = {}
        # Store waypoints by waypoint_id
        self.waypoints: typing.Dict[str, Waypoint] = {}
        # Store commands in a list (order might matter)
        self.commands: typing.List[Command] = []

        if start_time >= end_time:
            print(f"Warning: Mission start time ({
                  start_time}) is not before end time ({end_time}).")

    def add_drone(self, drone: Drone):
        """Adds a drone to the mission."""
        if drone.drone_id in self.drones:
            print(f"Warning: Drone with ID '{
                  drone.drone_id}' already exists, replacing.")
        self.drones[drone.drone_id] = drone

    def add_waypoint(self, waypoint: Waypoint):
        """Adds a waypoint to the mission."""
        if waypoint.waypoint_id in self.waypoints:
            print(f"Warning: Waypoint with ID '{
                  waypoint.waypoint_id}' already exists, replacing.")
        self.waypoints[waypoint.waypoint_id] = waypoint

    def add_command(self, command: Command):
        """Adds a command to the mission's command list."""
        # Optional: Validate if the command's drone_id exists in the mission
        if command.drone_id not in self.drones:
            print(f"Warning: Command '{
                  command.command_id}' targets non-existent drone '{command.drone_id}'.")

        # Optional: If it's a MovementCommand, validate if the goto_waypoint_id exists
        if isinstance(command, MovementCommand):
            if command.goto_waypoint_id not in self.waypoints:
                print(f"Warning: Movement command '{
                      command.command_id}' targets non-existent waypoint '{command.goto_waypoint_id}'.")

        self.commands.append(command)

    def __repr__(self):
        return (f"Mission(name='{self.name}', start={self.start_time}, end={self.end_time}, "
                f"{len(self.drones)} drones, {len(self.waypoints)} waypoints, {len(self.commands)} commands)")

    # Simple simulation step - moves drones towards their current waypoint command
    def simulate_step(self, step_interval_seconds: float):
        for drone_id, drone in self.drones.items():
            # Pass the full list of mission commands to the drone method
            command = drone.get_current_command(self.commands)

            if isinstance(command, MovementCommand):
                target_waypoint_id = command.goto_waypoint_id
                target_waypoint = self.waypoints.get(target_waypoint_id)

                if target_waypoint:
                    target_loc = target_waypoint.location
                    current_loc = drone.current_location

                    # Simple linear movement towards the target
                    vec_lat = target_loc.latitude - current_loc.latitude
                    vec_lon = target_loc.longitude - current_loc.longitude
                    vec_alt = target_loc.altitude - current_loc.altitude

                    distance_3d = math.sqrt(
                        vec_lat**2 + vec_lon**2 + vec_alt**2)

                    # Calculate step size in terms of coordinate change
                    # Distance covered in this interval
                    step_distance = drone.velocity * step_interval_seconds
                    step_fraction = 0.05  # Move 5% of the remaining distance towards the target per interval

                    if distance_3d > 1e-6:  # Avoid division by zero or tiny movements near target
                        # Don't overshoot
                        move_fraction = min(step_fraction, 1.0)

                        # Update position using the fraction
                        new_lat = current_loc.latitude + vec_lat * move_fraction
                        new_lon = current_loc.longitude + vec_lon * move_fraction
                        new_alt = current_loc.altitude + vec_alt * move_fraction

                        drone.current_location = Location(
                            new_lat, new_lon, new_alt)

                        distance_to_target_after_move = math.sqrt(
                            (target_loc.latitude - drone.current_location.latitude)**2 +
                            (target_loc.longitude - drone.current_location.longitude)**2 +
                            (target_loc.altitude - drone.current_location.altitude)**2
                        )

                        # Example threshold
                        if distance_to_target_after_move < (step_fraction * distance_3d * 0.5) or distance_to_target_after_move < 1e-4:
                            drone.current_location = target_loc  # Snap to the target location
                            drone.advance_command()  # Move to the next command
                            print(f"Drone {drone.drone_id} reached waypoint {
                                  target_waypoint_id}, advancing command.")
                            # Process the next command immediately
                            self.simulate_step(step_interval_seconds)

            elif isinstance(command, InternalCommand):
                print(f"Drone {drone.drone_id} is performing internal command {
                      command.command_id}. (Completing immediately in demo)")
                drone.advance_command()
                self.simulate_step(step_interval_seconds)

            elif command is None:
                pass


# --- RouteView Class (Dash Application) ---

class RouteView:
    # Class variable to hold the Mission instance
    # This allows the callback function to access the shared mission data
    # Using | None is fine if your Python is 3.10+, otherwise use typing.Optional
    mission: Mission | None = None

    def __init__(self, mission: Mission, name=__name__):
        if not isinstance(mission, Mission):
            raise TypeError(
                "RouteView must be initialized with a Mission object.")
        RouteView.mission = mission  # Store the mission instance in the class variable

        self.app = dash.Dash(name)
        # Get interval from the dcc.Interval component definition
        # A better approach is to define it once and use the variable here and in the layout
        # self.interval_seconds = 0.5 # Example value
        self.setup_layout()
        self.interval_seconds = 0.5  # Assume 0.5 seconds for simulation step
        self.setup_callbacks()

    def setup_layout(self):
        # Define the app layout
        self.app.layout = html.Div([
            html.H1(
                f"Live Map: Mission - {RouteView.mission.name if RouteView.mission else 'No Mission Loaded'}"),
            # Added style for full width and height
            dcc.Graph(id='live-map',
                      style={'width': '100%', 'height': '90vh'}),
            dcc.Interval(
                id='interval-component',
                interval=500,  # milliseconds (0.5 seconds)
                n_intervals=0
            )
        ])

    @staticmethod
    def lerp(v0, v1, t):
        """Linear interpolation - not used directly for movement here, but kept from previous code."""
        t = max(0.0, min(1.0, t))
        return (1 - t) * v0 + t * v1

    def setup_callbacks(self):
        # Define and register the callback function
        @self.app.callback(Output('live-map', 'figure'),
                           Input('interval-component', 'n_intervals'))
        def update_map(n):
            # Access the shared Mission instance from the class variable
            mission = RouteView.mission
            if mission is None:
                # Return an empty figure or a loading message if no mission is loaded
                return go.Figure()

            # --- Simulation Step ---
            mission.simulate_step(self.interval_seconds)
            # --- End Simulation Step ---

            current_positions = []

            # Get current positions from the Mission's Drones
            for drone_id, drone in mission.drones.items():
                current_positions.append({
                    'latitude': drone.current_location.latitude,
                    'longitude': drone.current_location.longitude,
                    'text': f"{drone.drone_id}<br>Batt: {drone.battery_lvl}%<br>Cmd: {drone.get_current_command(mission.commands)}"
                })

            df = pd.DataFrame(current_positions)

            # Create the map figure
            fig = go.Figure()

            # Add Drone markers
            if not df.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=df['latitude'],
                    lon=df['longitude'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(
                        size=10,  # Slightly larger markers
                        color='red'  # Different color for drones
                    ),
                    text=df['text'],
                    hoverinfo='text',
                    name='Drones'  # Add a trace name for legend
                ))

            # Optionally add waypoints to the map
            waypoint_lats = [
                wp.location.latitude for wp in mission.waypoints.values()]
            waypoint_lons = [
                wp.location.longitude for wp in mission.waypoints.values()]
            waypoint_names = [wp.name for wp in mission.waypoints.values()]

            if waypoint_lats:  # Only add if there are waypoints
                fig.add_trace(go.Scattermapbox(
                    lat=waypoint_lats,
                    lon=waypoint_lons,
                    mode='markers',
                    marker=go.scattermapbox.Marker(
                        size=8,
                        color='green',  # Waypoints in green
                        # Use a circle symbol for waypoints (star might not render universally)
                        symbol='circle'
                    ),
                    text=waypoint_names,
                    hoverinfo='text',
                    name='Waypoints'  # Add a trace name for legend
                ))

            all_lats = [pos['latitude']
                        for pos in current_positions] + waypoint_lats
            all_lons = [pos['longitude']
                        for pos in current_positions] + waypoint_lons

            if all_lats:  # Center on the mean of all points if any exist
                map_center_lat = np.mean(all_lats)
                map_center_lon = np.mean(all_lons)
            else:
                map_center_lat = 55.405  # Default Odenseish center
                map_center_lon = 10.40  # Default Odenseish center

            zoom_level = 11  # Start with a reasonable zoom

            if all_lats and len(all_lats) > 1:
                lat_range = max(all_lats) - min(all_lats)
                lon_range = max(all_lons) - min(all_lons)

                if lat_range > 0.0 or lon_range > 0.0:
                    degree_range = max(lat_range, lon_range)
                    # Adjust constants as needed
                    zoom_level = 12 - np.log2(max(degree_range, 0.01) / 0.1)
                    # Clamp zoom to a reasonable range
                    zoom_level = max(1, min(16, zoom_level))

            fig.update_layout(
                mapbox=dict(
                    bearing=0,
                    center=go.layout.mapbox.Center(
                        lat=map_center_lat,
                        lon=map_center_lon
                    ),
                    pitch=0,
                    zoom=zoom_level,
                    # or 'carto-positron', 'stamen-terrain', etc.
                    style='open-street-map'
                ),
                margin={"r": 0, "t": 30, "l": 0, "b": 0},
                # Add a title to the map figure itself
                title=f"Drone Locations - Mission: {mission.name}",
                # Add legend
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            return fig

    def run(self):
        # Method to run the Dash application
        self.app.run(debug=True)


# --- Example Usage ---
if __name__ == '__main__':
    # Create locations for known places in Odense, Denmark
    # Coordinates are approximate and might represent the general area or a specific entrance.
    loc_sdu = Location(latitude=55.3670, longitude=10.4271,
                       altitude=50.0)  # Near main SDU campus
    # Near the old OUH site / new construction area
    loc_ouh = Location(latitude=55.3853, longitude=10.3694, altitude=45.0)
    # Odense Banegård Center
    loc_banegaard = Location(
        latitude=55.4017, longitude=10.3871, altitude=15.0)

    # Create a mission
    current_time = int(time.time())
    mission_duration = 7200  # 2 hours
    my_mission = Mission(name="Odense Survey 2.0",
                         start_time=current_time,
                         end_time=current_time + mission_duration)

    # Create drones (example starting positions)
    drone1 = Drone(drone_id="drone_alpha", velocity=5.0,
                   battery_lvl=95, current_location=loc_sdu)  # Velocity in m/s
    drone2 = Drone(drone_id="drone_beta", velocity=3.5,
                   battery_lvl=80, current_location=loc_banegaard)

    # Create waypoints
    waypoint_sdu = Waypoint(waypoint_id="wp_sdu",
                            name="SDU Campus", location=loc_sdu)
    waypoint_ouh = Waypoint(waypoint_id="wp_ouh",
                            name="OUH Hospital", location=loc_ouh)
    waypoint_banegaard = Waypoint(
        waypoint_id="wp_banegaard", name="Train Station", location=loc_banegaard)

    # Add components to the mission
    my_mission.add_drone(drone1)
    my_mission.add_drone(drone2)
    my_mission.add_waypoint(waypoint_sdu)
    my_mission.add_waypoint(waypoint_ouh)
    my_mission.add_waypoint(waypoint_banegaard)

    # Create commands and add them to the mission (example sequence)
    # Drone alpha: SDU -> OUH -> Banegaard
    my_mission.add_command(MovementCommand(
        command_id="alpha_1", drone_id="drone_alpha", direction=0, turn="none", goto_waypoint_id="wp_ouh"))
    my_mission.add_command(MovementCommand(command_id="alpha_2", drone_id="drone_alpha",
                           direction=0, turn="none", goto_waypoint_id="wp_banegaard"))
    my_mission.add_command(InternalCommand(command_id="alpha_process",
                           drone_id="drone_alpha", is_processing=True))  # Add an internal command

    # Drone beta: Banegaard -> SDU -> Calibrate -> OUH
    my_mission.add_command(MovementCommand(
        command_id="beta_1", drone_id="drone_beta", direction=0, turn="none", goto_waypoint_id="wp_sdu"))
    my_mission.add_command(InternalCommand(
        command_id="beta_calibrate", drone_id="drone_beta", is_calibrating=True))
    my_mission.add_command(MovementCommand(
        command_id="beta_2", drone_id="drone_beta", direction=0, turn="none", goto_waypoint_id="wp_ouh"))

    # Create and run the RouteView with the mission
    route_app = RouteView(mission=my_mission)
    route_app.run()
