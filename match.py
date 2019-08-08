import datetime

class match:
    def __init__(self, match_id, game_mode, map, start_time, duration, telemetry_url):
        self.match_id = match_id
        self.teams = []
        self.game_mode = game_mode
        self.map = map
        self.start_time = start_time
        self.duration = duration
        self.telemetry_url = telemetry_url
        self.drop_flight_start = (None, None)  # Two dicts for start and end X/Y/Z coordinates

class team:
    def __init__(self, team_id):
        self.team_id = team_id
        self.player = []

class player:
    def __init__(self, member_id, username):
        self.member_id = member_id
        self.username = username
        self.plane_exit_loc = None
        self.parachute_land_loc = None
        self.kills = None  # Number of kills?  Or other players that were killed by this one?