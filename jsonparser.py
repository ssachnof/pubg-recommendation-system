import argparse
import multiprocessing
import pickle
import os
from threading import Thread

import matplotlib.pyplot as plt
from imageio import imread
import numpy as np
import pandas as pd
import math
import logging

from sklearn.externals import joblib

import downloader
import recommender as rec
import sklearn
import sklearn.preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.neighbors import KNeighborsClassifier

# TODO: remove global...
CM_TO_KM = 100000  # CM in a KM
MAX_MAP_SIZE = 8  # In KM


# search for specific attributes
# player_id :   players account id (eg. 'account.*****')
# start_range : starting time (eg. 'YYYY-MM-DDTHH:MM:SS.SSSZ')
# end_range:    ending time
# event_type:   list of type of event (eg. ['LogParachutepiLanding'])
def search(json_object, player_id=None, start_range=None, end_range=None, event_type=None):
    events = []
    i = 0
    for entry in json_object:
        if (is_event(entry, event_type) and
                in_range(entry['_D'], start_range, end_range) and
                has_player(entry, player_id)):
            events.append(entry)
    return events


def in_range(time, start, end):
    if start is None and end is None:
        return True
    elif start is None:
        return is_after(end, time)
    elif end is None:
        return is_after(time, start)
    else:
        return is_after(time, start) and is_after(end, time)


def is_after(time1, time2):
    t_index = time1.find('T')
    date1 = time1[:t_index].encode('ascii', 'ignore')
    date1 = date1.split('-')
    time1 = time1[t_index + 1:][:-1].encode('ascii', 'ignore')
    time1 = time1.split(':')

    t_index = time2.find('T')
    date2 = time2[:t_index].encode('ascii', 'ignore')
    date2 = date2.split('-')
    time2 = time2[t_index + 1:][:-1].encode('ascii', 'ignore')
    time2 = time2.split(':')

    equals = True

    for x in range(3):
        if int(date1[x]) > int(date2[x]):
            return True
    for x in range(2):
        if int(time1[x]) > int(time2[x]):
            return True

    if float(time1[2]) > float(time2[2]):
        return True

    if time1 != time2:
        return False

    return equals


def has_player(event, player_id):
    if player_id is None:
        return True

    for key in ['character', 'attacker', 'victim', 'assistant', 'reviver']:
        if key in event.keys():
            if event[key]['accountId'] == player_id:
                return True
        else:
            return False
    return False


def is_event(event, event_type):
    if event_type is None:
        return True
    else:
        return event['_T'] in event_type


# Return the dict from the pickle file name
def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        telemetry_data = pickle.load(f)
    f.close()
    return telemetry_data


# Get the location the plane starts in (not quite sure it's useful yet...)
def get_plane_start(telemetry):
    start_loc = None
    for log_entry in telemetry:
        if log_entry["_T"] == "LogMatchStart":
            # All players start at the exact same location, so we only need first
            start_loc = log_entry["characters"][0]['location']
    return start_loc


# Get the map name from telemetry
def get_map(telemetry):
    for log_entry in telemetry:
        if log_entry["_T"] == "LogMatchStart":
            return log_entry["mapName"]
    return None


# Get the result for each player in the match from the telemetry data
# Return the results as a list of dicts:
#       [{'name' : playerName, 'ranking': ranking},
#        ...]
def get_rankings(telemetry):
    results = []
    for log_entry in telemetry:
        if log_entry["_T"] == "LogMatchEnd":
            for character in log_entry['characters']:
                results.append({'name': character['name'],
                                'ranking': character['ranking']})
    return results


# Get the location of each player's landing from the telemetry data
# Return as a dict:
#           {playerName: [x, y],
#            ...}
def get_all_landings(telemetry):
    landings = {}
    for log_entry in telemetry:
        if log_entry["_T"] == "LogParachuteLanding":
            landing_loc = log_entry['character']['location']
            x = landing_loc['x']
            y = landing_loc['y']
            landings[log_entry['character']['name']] = [x, y]
    return landings


# Returns the first  and last locations someone jumped out of the plane
def get_flight_data(telemetry):
    first_coordinate = None  # First player exit event from plane
    current_coordinate = None  # Last player exist even from plane
    for log_entry in telemetry:
        if log_entry.get("_T") == "LogVehicleLeave" \
                and log_entry.get("vehicle").get("vehicleId") == "DummyTransportAircraft_C":
            current_coordinate = log_entry.get("character").get("location")
            if first_coordinate is None:
                first_coordinate = current_coordinate
    return first_coordinate, current_coordinate


# Returns the unit vector for the flight path given the first and last location someone exited plane
def get_flight_vector(first_drop, last_drop):
    flight_vector = np.array([last_drop['x'] - first_drop['x'], last_drop['y'] - first_drop['y']])
    vector_length = np.linalg.norm(flight_vector)
    flight_vector_norm = flight_vector / vector_length
    return flight_vector_norm


# Returns the angle between the two flight vectors u and v (squared)
# Note: u and v must be unit vectors already
def flight_diff(u, v):
    return np.arccos(np.clip(np.dot(u, v), -1, 1)) ** 2


# Given a flight path (unit vectorized), get the direction that it is closest to
# from one of the eight recognized cardinal directions
def get_flight_category(flight_vector):
    # Set up unit vectors for the 8 directions to classify flight paths as
    between_mag = 1 / math.sqrt(2)
    nn = [0, 1]
    ne = [between_mag, between_mag]
    ee = [1, 0]
    se = [between_mag, -between_mag]
    ss = [0, -1]
    sw = [-between_mag, -between_mag]
    ww = [-1, 0]
    nw = [-between_mag, between_mag]

    # Put em in a DataFrame for processing
    directions = np.array([nn, ne, ee, se, ss, sw, ww, nw])
    dirs_df = pd.DataFrame(directions)

    # Combine the x, y component columns, add direction names, and get rid of the excess
    dirs_df['direction_vec'] = list(zip(dirs_df[0], dirs_df[1]))
    dirs_df['direction'] = np.array(['nn', 'ne', 'ee', 'se', 'ss', 'sw', 'ww', 'nw'])
    dirs_df.drop([0, 1], axis=1, inplace=True)

    # Calculate the angle between the flight_vector and each of the 8 directions
    # and store it under a new column, 'angle_from_path'
    dirs_df['angle_from_path'] = dirs_df['direction_vec'].apply(flight_diff, args=(flight_vector,))

    # And return the direction (string) where the angle between the flight_vector
    # and the direction is minimized (i.e. the closest direction to the flight_vector)
    return dirs_df['direction'].loc[dirs_df['angle_from_path'].idxmin()]


def get_flight_cat_from_telemetry(telemetry):
    first, last = get_flight_data(telemetry)
    if first is not None and last is not None:
        vec = get_flight_vector(first, last)
        return get_flight_category(vec)
    else:
        return None


# Convert raw (x, y) coordinates to the category
# Maps are divided into a 20x20 grid of square_size x square_size blocks
# where each square can be represented by a letter for the x and y position of that
# square in the map, e.g. AA for the square containing (0, 0)
def get_loc_cat(x, y, map_dim):
    square_size = 0.05 * map_dim
    x_ = x // square_size
    y_ = y // square_size

    # add the number square along each access to 'A's ascii code and return the characters
    # as the category
    return chr(65 + int(x_)) + chr(65 + int(y_))


# Plot the drop locations of each player (in blue), with opacity in relation to their rank in that match
# (more opaque = lower rank), along with the location the first person left the plane (in green)
# and the last person to leave the plane (in red)
def display_drop_locations(telemetry, fig, fig_x, fig_y, fig_num, match_num):
    landings = get_all_landings(telemetry)
    rankings = get_rankings(telemetry)
    map_name = get_map(telemetry)

    # Set up plot scale
    if map_name == "Savage_Main":  # 4km map
        x_max = MAX_MAP_SIZE * (1 / 2)
        y_max = x_max
        map_img = imread("savage.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 4.0, 0.0, 4.0])
    elif map_name == "Erangel_Main":  # 8km map
        x_max = MAX_MAP_SIZE
        y_max = MAX_MAP_SIZE
        map_img = imread("erangel.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 8.0, 0.0, 8.0])
    elif map_name == "Desert_Main":  # 8km map
        x_max = MAX_MAP_SIZE
        y_max = x_max
        map_img = imread("miramar.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 8.0, 0.0, 8.0])
    elif map_name == "DihorOtok_Main":  # 6km map
        x_max = MAX_MAP_SIZE * (3 / 4)
        y_max = x_max
        map_img = imread("vikendi.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 6.0, 0.0, 6.0])

    first_launch, last_launch = get_flight_data(telemetry)

    if first_launch is not None:
        launch_x = [first_launch['x'], last_launch['x']]
        launch_y = [first_launch['y'], last_launch['y']]

        ax = fig.add_subplot(fig_x, fig_y, fig_num)

        # plot first and last jump locations
        ax.scatter(launch_x[0] / CM_TO_KM, launch_y[0] / CM_TO_KM, s=100,
                   color='green', edgecolors='black', zorder=1)
        ax.scatter(launch_x[1] / CM_TO_KM, launch_y[1] / CM_TO_KM, s=100,
                   color='red', edgecolors='black', zorder=1)

        # plot line between them
        ax.plot([x_ / CM_TO_KM for x_ in launch_x],
                [y_ / CM_TO_KM for y_ in launch_y], 'grey', linestyle='--', marker='', zorder=1)

        # plot each player according to their ranking
        for ranking in rankings:
            landing_loc = landings[ranking['name']]
            # print("Player {} landing at position\t ({}, {}) and ended up rank : {}".format(ranking['name'],
            #                                                                           landing_loc[0],
            #                                                                          landing_loc[1],
            #                                                                          ranking['ranking']))
            if ranking['ranking'] == 1:
                ax.scatter(landing_loc[0] / CM_TO_KM, landing_loc[1] / CM_TO_KM,
                           color='yellow', edgecolors='black', zorder=1)
            else:
                ax.scatter(landing_loc[0] / CM_TO_KM, landing_loc[1] / CM_TO_KM,
                           color='blue', alpha=1 / ranking['ranking'], zorder=1)
        plt.ylim(0, y_max)
        plt.xlim(0, x_max)
        plt.xlabel('km')
        plt.ylabel('km')
        plt.title(map_name)
        plt.savefig('./match_landings/match_{}.png'.format(match_num))
        plt.show()
    else:
        logging.error("Could not get launch data")


def drop_data_worker(filename_queue, drop_data):
    logging.info("Parser thread active")

    while not filename_queue.empty():
        file_path = filename_queue.get()
        # If the filesize is greater than 0, ie there is actual data in it
        if os.path.getsize(file_path) > 0:
            try:
                telemetry = load_pickle(file_path)
                #logging.debug("Loaded match file %s, approximately %i remaining in queue", file_path, filename_queue.qsize())
            except EOFError:
                logging.error("Match file %s terminated unexpectedly, skipping", file_path)
                continue  # Skip processing files that terminate early
            first, last = get_flight_data(telemetry)
            map_name = get_map(telemetry)

            # Set map_size (in cm, like player locations)
            if map_name == "Savage_Main":  # 4km map
                map_size = 400000
            elif map_name == "Erangel_Main":  # 8km map
                map_size = 800000
            elif map_name == "Desert_Main":  # 8km map
                map_size = 800000
            elif map_name == "DihorOtok_Main":  # 6km map
                map_size = 600000
            else:
                continue

            # Get the flight direction
            if first is not None:
                flight_vec = get_flight_vector(first, last)
                direction = get_flight_category(flight_vec)

            # Get the landings
            landing_locs = get_all_landings(telemetry)
            rankings = get_rankings(telemetry)

            for player, loc in landing_locs.items():
                # For each player who dropped, get their rank from the rankings array
                loc_category = get_loc_cat(loc[0], loc[1], map_size)
                for ranking in rankings:
                    if player in ranking.values():
                        player_rank = ranking['ranking']

                drop_data.append({'player': player,
                                  'drop_loc_raw': loc,
                                  'drop_loc_cat': loc_category,
                                  'rank': player_rank,
                                  'flight_path': direction,
                                  'map': map_name})
                # print("{} ==> {}".format(flight_vec, dir))
            # display_drop_locations(telemetry, plt.figure(), 1, 1, 1, match_num)


def build_drop_data(telemetry_files, data_dir, download_threads):
    drop_data = []
    threads = []
    filename_queue = multiprocessing.Queue()

    for match_num in range(0, len(telemetry_files)):
        filename_queue.put(data_dir + telemetry_files[match_num])

    for i in range(download_threads):
        new_thread = Thread(target=drop_data_worker, args=(filename_queue, drop_data))
        new_thread.daemon = True
        new_thread.start()
        logging.debug("Starting thread %i", i)
        threads.append(new_thread)

    for i in range(download_threads):
        threads[i].join()

    return pd.DataFrame(drop_data)


# Get safe zone and poison zone states (location and radius) throughout the game
# Returns dataframe for each time & states
# columns: ['_D', 'safetyZonePosition_x', 'safetyZonePosition_y', 'safetyZoneRadius', ...]
def get_zone_states(json_object):
    logged_game_states = search(json_object, None, None, None, ['LogGameStatePeriodic'])
    all_states = []
    iterated_states = []
    for gameState in logged_game_states:
        timestamp = gameState['common']['isGame']
        if timestamp not in iterated_states:
            state = gameState['gameState']
            new_state_object = {k: state[k] for k in ('safetyZoneRadius',
                                                      'poisonGasWarningRadius')}
            safe_position = state['safetyZonePosition']
            poison_position = state['poisonGasWarningPosition']

            new_state_object['safetyZonePosition_x'] = safe_position['x']
            new_state_object['safetyZonePosition_y'] = safe_position['y']
            new_state_object['isGame'] = timestamp
            new_state_object['poisonGasWarningPosition_x'] = poison_position['x']
            new_state_object['poisonGasWarningPosition_y'] = poison_position['y']
            all_states.append(new_state_object)
            iterated_states.append(timestamp)
    df = pd.DataFrame(all_states)
    return df


# Get items picked up (house, crate, loot, etc) and by whom
# Returns dataframe for each pickup log
# columns: ['character_accountId', 'character_name', 'item_category', ...]
def get_item_pickup(json_object):
    item_pickups = search(json_object,
                          None,
                          None,
                          None,
                          ['LogItemPickup',
                           'LogItemPickupFromCarepackage',
                           'LogItemPickupFromLootBox'])
    parsed_data = []
    for log in item_pickups:
        char = log['character']
        item = log['item']
        pickup_state = {
            '_D': log['_D'],
            'character_accountId': char['accountId'],
            'character_name': char['name'],
            'item_category': item['category'],
            'item_subCategory': item['subCategory'],
            'item_Id': item['itemId']}
        parsed_data.append(pickup_state)
    return pd.DataFrame(parsed_data)


# Returns a list of DataFrames where each DataFrame contains all of the drop data for a given map and flight path
def get_drop_data(data_dir, threads):
    match_files = []
    telemetry_files = []

    logging.info("Scanning for match and telemetry files in %s to parse", data_dir)
    for file in os.listdir(data_dir):
        if "_match" in file:
            logging.debug("Match file %s found, adding as match", file)
            match_files.append(file)
        elif "_telemetry" in file:
            logging.debug("Telemetry file %s found, adding as match", file)
            telemetry_files.append(file)

    # Get aggregate data
    drop_data = build_drop_data(telemetry_files, data_dir, threads)

    # Split by map and flight path
    all_data = []
    map_data_li = split_drop_data_by_map(drop_data)
    for map_df in map_data_li:
        if map_df.iloc[0]['map'] == "Savage_Main":
            flight_data_li = [map_df]
        else:
            flight_data_li = split_drop_data_by_flight_path(map_df)
        all_data.extend(flight_data_li)

    return all_data


# Split the DataFrame containing all of the drop data into separate DataFrames for each map
def split_drop_data_by_map(drop_data):
    map_data = []
    for game_map in drop_data['map'].unique():
        map_data.append(drop_data[drop_data['map'] == game_map])
    if not map_data:
        logging.error("Got empty result when attempting to split data by map")
    return map_data


# Split the drop data (assumed to already be split by map) by flight path
def split_drop_data_by_flight_path(drop_data):
    map_data = []
    flight_data = []
    for flight in drop_data['flight_path'].unique():
        flight_data.append(drop_data[drop_data['flight_path'] == flight])
    if not flight_data:
        logging.error("Got empty result when attempting to split data by flight path")
    return flight_data


def success_category(x):
    return x // 20


def preprocess_data(df):
    labelencoder_x = sklearn.preprocessing.LabelEncoder()
    x = df.iloc[:, :].values
    df['flight_path'] = labelencoder_x.fit_transform(x[:, 1])
    df['map'] = labelencoder_x.fit_transform(x[:, 0])
    df['success_category'] = labelencoder_x.fit_transform(x[:, -1])
    return df


def main():
    parser = argparse.ArgumentParser("Parses PUBG match data")
    parser.add_argument('--downloaddir')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()
    download_directory = "data/"
    if args.downloaddir is not None:
        download_directory = args.downloaddir
    threads = os.cpu_count()
    if args.threads is not None:
        threads = int(args.threads)
    downloader.setup_logging(args.debug)
    logging.info("Initialized parser")

    drop_data = get_drop_data(download_directory, threads)
    logging.info("Drop data loaded")


    # for i in range(len(drop_data)):
    #     df = drop_data[i]
    #     df = df[df["rank"] <= 10]
    #     print(df)
    #     df['x_drop_loc_raw'] = df['drop_loc_raw'].apply(lambda x: x[0])
    #     df['y_drop_loc_raw'] = df['drop_loc_raw'].apply(lambda x: x[1])
    #     drop_data[i] = df
    #
    # for i in range(len(drop_data)):
    #     df = drop_data[i]
    #     df = df.drop(columns=['drop_loc_raw'])
    #
    # for i in range(len(drop_data)):
    #     df = drop_data[i]
    #     df['success_category'] = df['rank'].apply(success_category)
    #     drop_data[i] = df
    # for i in range(len(drop_data)):
    #     df = drop_data[i]
    #     df = df.drop(columns=["drop_loc_cat", "drop_loc_raw", "player", "rank"])
    #     drop_data[i] = df



            # temp = preprocess_data(drop_data[0])
    # for df in drop_data:
    #     print(df)
    #     rec.train_model(preprocess_data(df), 6)
    #     print("success!!!!")
    #rec.train_model(drop_data, 7)
    a = np.array(['a', 'b', 'f', 'd'])
    b = np.array(['f', 'b', 'e', 'd'])
    c = pd.DataFrame(np.array([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 2]]), columns=['a', 'b', 'c'])


if __name__ == "__main__":
    main()
