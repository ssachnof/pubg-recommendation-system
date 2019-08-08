import jsonparser
import pandas as pd
from pandas.io.json import json_normalize


def round_raw(raw):
    """
    Convert raw position value to halfway in the nearest 100m interval (Center of the grid square for that interval)
        e.g. position (126903, 341005) == (1.27km, 3.41km) -> (125000, 345000) == (1.25km, 3.45km)

    :param raw: raw value of coordinate (in cm)
    :return: rounded value to the nearest .05km
    """
    return int((raw // 10000) * 10000 + 5000)


def get_player_paths(telemetry):
    """
    Get all logged locations of the players from the telemetry data, along with the character name
    (for joining purposes with rank) and the game state (for joining purposes with the zone data)

    :param telemetry: json object of all the telemetry events for a given match
    :return: DataFrame with columns:

            [game_state, name, x, y]

            where each row represents a log of an individual player position.
            Players have multiple logs per game, i.e. multiple rows in the DataFrame
    """
    telemetry_df = json_normalize(telemetry)
    positions_df = (telemetry_df[telemetry_df['_T'] == 'LogPlayerPosition']
        [['common.isGame', 'character.name', 'character.location.x', 'character.location.y']])
    positions_df.columns = positions_df.columns.map({
        'common.isGame': 'game_state',
        'character.name': 'name',
        'character.location.x': 'x',
        'character.location.y': 'y'
    })

    rankings = jsonparser.get_rankings(telemetry)
    rankings_df = pd.DataFrame(rankings)
    return positions_df.set_index("name").join(rankings_df.set_index("name")).dropna().reset_index()


def join_player_and_zone(player_paths, zone_info):
    """
    Join a DataFrame with player location info and a DataFrame with the zones location info

    :param player_paths: DataFrame containing information about player locations throughout the game
    :param zone_info: DataFrame containing information about the poison gas zone and safe zone throughout the game
    :return: DataFrame indexed on the game state (0.0, 0.1, 0.5, 1.0,...) with columns:

            [gameState, name, x_, y_, ranking, poisonGasWarningPosition_x, poisonGasWarningPosition_y,
             poisonGasWarningRadius, safetyZonePosition_x, safetyZonePosition_y, safetyZoneRadius]

            where x_ and y_ are the rounded values of the raw locations.
            Each row represents a single player's location at a given time.
            Poison gas and safety zone values may be NaN for logs prior to match start.
    """
    player_paths = player_paths.dropna().reset_index()
    player_paths['x_'] = player_paths['x'].apply(round_raw)
    player_paths['y_'] = player_paths['y'].apply(round_raw)
    player_paths.drop(['x', 'y'], axis=1, inplace=True)
    zone_info.drop(['safetyZonePosition_x', 'safetyZonePosition_y', 'safetyZoneRadius'], axis=1, inplace=True)

    joined_df = (player_paths.set_index("game_state").join(zone_info.set_index("isGame"))
                 .drop("index", axis=1).reset_index())
    joined_df.columns = joined_df.columns.map({"index" : "gameState",
                                               "name": "name",
                                               "x_": "x",
                                               "y_": "y",
                                               "ranking": "ranking",
                                               "poisonGasWarningPosition_x": "safe_x",
                                               "poisonGasWarningPosition_y": "safe_y",
                                               "poisonGasWarningRadius": "safe_r"
                                               })

    return joined_df

