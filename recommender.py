from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

import jsonparser
import player_info
import logging
import os
import downloader
import data_visualization

import pickle
from math import sqrt, pi, cos, sin, atan2
from random import random, randint
import pandas as pd
import numpy as np
import constants
import random


def in_zone(x, y, zone_x, zone_y, zone_r):
    """ Given (x, y) of a position, return True if it is within the zone boundaries
        and False if it is not
    """
    dist_to_center = sqrt((x - zone_x)**2 + (y - zone_y)**2)
    return dist_to_center < zone_r


def gen_new_safezone(curr_x, curr_y, curr_r, rad_decrease):
    """
    Given the current safe zone properties and the proportion to decrease the next one by, generate the next safe zone

    :param curr_x: current x coordinate of the safe zone center
    :param curr_y: current y coordinate of the safe zone center
    :param curr_r: current radius of the safe zone
    :param rad_decrease: the ratio to decrease the circle radius by. Typically 0.5
    :return: x, y and radius of the new safe zone
    """
    new_r = curr_r * rad_decrease

    # Get random radius to new point within new_r
    r_ = new_r * sqrt(random())
    # Get random angle
    theta = random() * 2 * pi

    new_x = curr_x + r_ * cos(theta)
    new_y = curr_y + r_ * sin(theta)

    return new_x, new_y, new_r


def get_closest_to_safezone(x, y, safe_x, safe_y, safe_r):
    """
    Get the point in the safe zone that is closest to the player location
    (Assumed that the player location is OUTSIDE the safe zone)

    :param x: player x coordinate
    :param y: player y coordinate
    :param safe_x: x coordinate of safe zone center
    :param safe_y: y coordinate of safe zone center
    :param safe_r: safe zone radius
    :return: the point (rounded like the player locations) closest to the safe zone
    """
    distance = sqrt((x - safe_x)**2 + (y - safe_y)**2)
    to_move = distance - safe_r
    angle = atan2(safe_y - y, safe_x - x)
    x_ = player_info.round_raw(x + to_move * cos(angle))
    y_ = player_info.round_raw(y + to_move * sin(angle))
    return x_, y_


def gen_candidate_locations(curr_x, curr_y, next_safe_x, next_safe_y, next_safe_r):
    """
    Given a player location and where the next safe zone is, calculate the neighboring locations to use
    as candidates for path generation. Only neighboring locations in the safe zone are considered.
     --> (return may be empty list)

    :param curr_x: player x coordinate
    :param curr_y: player y coordinate
    :param next_safe_x: safe zone x coordinate
    :param next_safe_y: safe zone y coordinate
    :param next_safe_r: safe zone radius
    :return: a list of [(x1, y1), (x2, y2), ...] of each neighboring location
    """
    candidates = []
    for x in range(curr_x - 10000, curr_x + 20000, 10000):
        for y in range(curr_y - 10000, curr_y + 20000, 10000):
            if in_zone(x, y, next_safe_x, next_safe_y, next_safe_r):
                candidates.append((x, y))
    return candidates


def get_next_loc(game_state, x, y, safe_x, safe_y, safe_r, model):
    """
    Given the current game_state, player location, safe zone location, and model, get the next location to add to path

    :param game_state: current game state
    :param x: player x coordinate
    :param y: player y coordinate
    :param safe_x: x coordinate of safe zone center
    :param safe_y: y coordinate of safe zone center
    :param safe_r: safe zone radius
    :param model: model used to predict whether locations will result in win or not
    :return:
    """
    candidates = gen_candidate_locations(x, y, safe_x, safe_y, safe_r)
    if len(candidates) == 0:    # No usual candidates were in the zone
        return get_closest_to_safezone(x, y, safe_x, safe_y, safe_r)

    winning_locs = []
    ranks = {0: [], 1: [], 2: [], 3: [], 4: []}
    for cand_x, cand_y in candidates:
        rank = predict_rank(game_state, cand_x, cand_y, safe_x, safe_y, safe_r, model)
        ranks[rank].append((cand_x, cand_y))

    if len(ranks[0]) > 0:
        print("NEXT LOC RANK 0")
        return ranks[0][randint(0, len(ranks[0]) - 1)]
    elif len(ranks[1]) > 0:
        print("NEXT LOC RANK 1")
        return ranks[1][randint(0, len(ranks[1]) - 1)]
    elif len(ranks[2]) > 0:
        print("NEXT LOC RANK 2")
        return ranks[2][randint(0, len(ranks[2]) - 1)]
    elif len(ranks[3]) > 0:
        print("NEXT LOC RANK 3")
        return ranks[3][randint(0, len(ranks[3]) - 1)]
    elif len(ranks[4]) > 0:
        print("NEX LOC RANK 4")
        return ranks[4][randint(0, len(ranks[4]) - 1)]
    else:
        return -1, -1


def predict_rank(game_state, x, y, safe_x, safe_y, safe_r, model):
    """
    Given information about a location, time, and where the safe zone is, predict whether
    the location is likely to result in a win or loss

    :param game_state: float representing the time of game: 0.5, 1.0... etc
    :param x: x coordinate of location to predict
    :param y: y coordinate of location to predict
    :param safe_x: x coordinate of the center of the safe zone
    :param safe_y: y coordinate of the center of the safe zone
    :param safe_r: radius of the safe zone
    :param model: the model to predict with
    :return: 1 if the location is predicted to be a winning location, 0 if it is not
    """
    predicted = model.predict(np.array([game_state, x, y, safe_x, safe_y, safe_r]).reshape(1, -1))
    return int(predicted[0].item())


def gen_path(drop_x, drop_y, possible_safe_zones, end_state, model):
    """
    Given a drop location, potential locations for the first safe zone, and a model, generate a path
    starting at the drop location.

    Path is generated by looking at all neighboring locations that are predicted to result in a win and selecting a
    random one as the next location. If there are no neighboring locations predicted to result in a win, the location
    does not change and the path ends. Otherwise path generation continues until the end_state (a game_state) is reached.
    For each game_state (0.5, 1, 1.5,...) there are two locations generated, and on each even game_state the safe zone
    is updated. This results in 4 locations for each safe zone, except for the first zone which only has 2 generated
    (+ the original drop location)

    :param drop_x: x coordinate of the drop location to start from
    :param drop_y: y coordinate of the drop location to start from
    :param possible_safe_zones: a DataFrame where the columns are:
                                        [x, y, radius]
                                and each row represents a possible first safe zone
    :param end_state: the game_state to generate paths up to. Typical values from observed games are in the range of
                        6.5 to 9.5
    :param model: the model to use for predicting whether a location will result in a win
    :return: a DataFrame with columns:
                    [x, y, game_state, safe_x, safe_y, safe_r]
             where the first row is the drop location and each subsequent row is the next location in the path
    """
    safe_zone = possible_safe_zones.sample(n=1)
    safe_x = safe_zone["x"].values[0].item()
    safe_y = safe_zone["y"].values[0].item()
    safe_r = safe_zone["radius"].values[0].item()

    curr_x = drop_x
    curr_y = drop_y
    path = list()

    game_state = 0.5
    path.append({"x": curr_x, "y": curr_y,
                 "game_state": game_state,
                 "safe_x": safe_x,
                 "safe_y": safe_y,
                 "safe_r": safe_r})

    print("SAFE ZONE STARTING AT {}, {} : {}".format(safe_x, safe_y, safe_r))

    # While the end_state has not been reached
    while game_state < end_state:

        # Get the next position to move
        curr_x, curr_y = get_next_loc(game_state, curr_x, curr_y, safe_x, safe_y, safe_r, model)

        if curr_x == -1 and curr_y == -1:   # No candidate locations were predicted to be winning locations, path ends
            game_state = end_state
            print("NO WINNING MOVES - YOU DIED")
        else:
            # Add to path
            path.append({"x": curr_x, "y": curr_y,
                         "game_state": game_state,
                         "safe_x": safe_x,
                         "safe_y": safe_y,
                         "safe_r": safe_r})

            game_state += 0.25

        # Update safe zone if the game_state is a whole number
        if int(game_state) == game_state:
            safe_x, safe_y, safe_r = gen_new_safezone(safe_x, safe_y, safe_r, 0.5)
            print("NEW SAFE ZONE AT {}, {} : {}".format(safe_x, safe_y, safe_r))
    return pd.DataFrame(path)


# note that I am assuming the target is in the last position of the dataframe
# additionally, I am assuming that the list has already been filtered(ie. we are only training on the top players)
# additionally, my current assumption is the data has already been transformed into non-categorical data
def train_model(df, max_k):
    x = df[['x_drop_loc_raw', 'y_drop_loc_raw']]
    y = df['success_category']
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
    best_score = 0.0
    best_model = None
    for k in range(1, max_k):
        scaler = StandardScaler()
        model = KNeighborsClassifier(n_neighbors=k)
        pipeline = Pipeline([('scaler', scaler),
                             ('fit', model)])
        score = cross_val_score(pipeline, x, y, cv=5, scoring='accuracy').mean()



        if score > best_score or best_model is None:
            best_score = score
            best_model = pipeline
        print("Best Accuracy Score: " + str(best_score))
    best_model.fit(x, y)
        # return best_model
    return best_model

# preprocess the dataframe
def preprocess_data(drop_data):
    drop_data = drop_data.dropna()
    drop_data = drop_data.drop(columns=['player'])  # probably don't need to include the player in the model
    drop_data = drop_data.drop(columns=['drop_loc_raw'])  # probably don't need to include the player in the model
    drop_data = drop_data.dropna()
    drop_data = drop_data[drop_data['rank'] <= 5]
    labelencoder_x = LabelEncoder()
    x = drop_data.iloc[:, :].values
    drop_data['flight_path'] = labelencoder_x.fit_transform(x[:, 1])
    drop_data['map'] = labelencoder_x.fit_transform(x[:, 2])
    drop_data = drop_data.drop(columns=['rank'])
    drop_data = drop_data[['flight_path', 'map', 'drop_loc_cat']]
    scaler = MinMaxScaler()
    drop_data.loc[:, :-1] = scaler.fit_transform(drop_data[drop_data.columns[:-1]])
    return drop_data


def tune_player_path_model(position_df, max_k):
    """ Get the optimal k value for predicting rank based on player position and the locations of the two zones

    :param position_df: Output of player_info.join_player_and_zone(...)
    :param max_k:       max K value to test
    :return:            the model that resulted in the highest accuracy when predicting rank
    """
    # Zone and player data
    x = position_df.drop(['name', 'ranking'], axis=1)

    # Player rank
    y = position_df['ranking']

    best_score = 0
    best_model = None

    # Hyperparameter Tuning
    for k in range(1, max_k):
        scaler = StandardScaler()
        model = KNeighborsClassifier(n_neighbors=k)
        pipeline = Pipeline([('scaler', scaler),
                             ('fit', model)])

        score = cross_val_score(pipeline,
                                x,
                                y,
                                cv=5, scoring='accuracy').mean()

        #print("\tacc: ", score)
        if score > best_score or best_model is None:
            best_score = score
            best_model = pipeline

    print("Best Accuracy Score: " + str(best_score))
    return best_model

def get_drop_data_by_map(drop_data):
    for i in range(len(drop_data)):
        df = drop_data[i]
        df['x_drop_loc_raw'] = df['drop_loc_raw'].apply(lambda x: x[0])
        df['y_drop_loc_raw'] = df['drop_loc_raw'].apply(lambda x: x[1])
        drop_data[i] = df

    for i in range(len(drop_data)):
        df = drop_data[i]
        df = df.drop(columns=['drop_loc_raw'])

    for i in range(len(drop_data)):
        df = drop_data[i]
        rank_range = df["rank"].max() - df["rank"].min()
        df["success_category"] = df["rank"].apply(ranking_to_bin, args=(rank_range,))
        drop_data[i] = df
    for i in range(len(drop_data)):
        df = drop_data[i]
        df = df.drop(columns=["drop_loc_cat", "drop_loc_raw", "player", "rank"])
        drop_data[i] = df
def ranking_to_bin(ranking, rank_range):
    rank_bin = (ranking - 1) // (rank_range // 5)
    if rank_bin == 5:   # Range doesn't divide into 5 evenly, so there will be a 6th bin, need to add to 5th instead
        rank_bin = 4
    return rank_bin


def get_map_data(telemetry_files):
    """
    Given a list of telemetry file names, extract the player location and safe zone info to aggregate by map and flight
    path

    :param telemetry_files: list of telemetry file names
    :return: dict: {(map_name, flight_path): DataFrame of locations with safe zone, ...}
    """
    map_data = dict()
    for i, telemetry_file in enumerate(telemetry_files):
        print("\tMatch {} of {}".format(i, len(telemetry_files)))
        telemetry = jsonparser.load_pickle(data_dir + telemetry_file)
        flight_cat = jsonparser.get_flight_cat_from_telemetry(telemetry)
        map_name = jsonparser.get_map(telemetry)

        if flight_cat is not None:
            print(map_name, " : ", flight_cat)
            player_loc_info = player_info.get_player_paths(telemetry)
            zone_info = jsonparser.getZoneStates(telemetry)
            combined = player_info.join_player_and_zone(player_loc_info, zone_info).dropna()
            rank_range = combined["ranking"].max() - combined["ranking"].min()
            combined["ranking"] = combined["ranking"].apply(ranking_to_bin, args=(rank_range,))
            print(combined["ranking"].value_counts())
            print("MAX STATE: ", combined['gameState'].max())

            if (map_name, flight_cat) not in map_data.keys( ):
                map_data[(map_name, flight_cat)] = []
            map_data[(map_name, flight_cat)].append(combined.dropna())

    for key, data in map_data.items():
        map_data[key] = pd.concat(data)

    return map_data


def train_models(map_data):
    """ Given the data for each map, train models for that data, fit them to the data, and pickle them

    :param map_data: dict: {(map_name, flight_path): DataFrame of player location, ...}
    :return:         dict: {(map_name, flight_path): DataFrame of models, ...}
    """
    models = dict()
    for key, data in map_data.items():
        print(key, " : ", len(data))
        optimal = tune_player_path_model(data, 15)

        data_x = data.drop(['name', 'ranking'], axis=1)
        data_y = data['ranking']
        optimal.fit(data_x, data_y)

        models[key] = optimal

        with open("./models/{}_{}-model.pickle".format(key[0], key[1]), "wb") as model_f:
            pickle.dump(optimal, model_f)
            model_f.close()

    return models

# get_drop_data_by_map should be called before this
def train_models_drop_locations(drop_data, max_k):
    '''
    writes trains and writes drop data dataframe to a file
    :param drop_data: array[dataframe] for all map, flight path drop data
    :return: void
    '''
    for i in range(len(drop_data)):
        df = drop_data[i]
        mapName = df.iloc[0]['map']
        flight_direction = df.iloc[0]['flight_path']
        if mapName == 'Savage_Main':
            filepath = "./drop_models/model_" + mapName + ".pkl"
        else:
            filepath = "./drop_models/model_" + mapName + "_" + flight_direction + ".pkl"
        model = train_model(df, max_k)
        logging.debug("SAVING MODEL TO PATH " + filepath)
        joblib.dump(model, filepath)


def get_map_constraints(mapName):
    '''

    :param mapName: String
    :return: tuple(:min_x: int, :max_x: int, :min_y: int, :max_y: int)
    '''
    if mapName == 'Desert_Main':
        return (constants.DESERT_MIN_X, constants.DESERT_MAX_X, constants.DESERT_MIN_Y, constants.DESERT_MAX_Y)
    elif mapName == 'Erangel_Main':
        return (constants.ERANGEL_MIN_X, constants.ERANGEL_MAX_X, constants.ERANGEL_MIN_Y, constants.ERANGEL_MAX_Y)
    elif mapName == 'DihorOtok_Main':
        return (constants.DIHOROTOK_MIN_X, constants.DIHOROTOK_MAX_X, constants.ERANGEL_MIN_Y, constants.ERANGEL_MAX_Y)
    else:
        assert(mapName == 'Savage_Main')
        return (constants.SAVAGE_MIN_X, constants.SAVAGE_MAX_X, constants.SAVAGE_MIN_Y, constants.SAVAGE_MAX_Y)

def get_drop_predictions(mapName, flight_path, model):
    '''

    :param mapName: String
    :param flight_path: String
    :param model: sklearn Pipeline
    :return: void

    writes to a csv file for the best predicted drop locations
    '''

    min_x, max_x, min_y, max_y = get_map_constraints(mapName)
    locations = {'x_location': [], 'y_location': []}
    # find the best drop locations
    for x in range(min_x, max_x + 1, constants.DROP_MAP_INCREMENT):
        for y in range(min_y, max_y + 1, constants.DROP_MAP_INCREMENT):
            locations['x_location'].append(x)
            locations['y_location'].append(y)

    locations = pd.DataFrame(locations)
    predicted_ranks = model.predict(locations)
    locations['predicted_rank'] = predicted_ranks
    best_predicted_rank = min(predicted_ranks)

    locations = locations[locations['predicted_rank'] == best_predicted_rank]

    # create a dataframe of the best drop locations and then write to a csv
    if mapName != "Savage_Main":
        csv_file_path = "./drop_locations/drop_" + mapName + "_" + flight_path + ".csv"
    else:
        csv_file_path = "./drop_locations/drop_" + mapName + ".csv"
    locations.to_csv(csv_file_path)
    print("predictions for flight_path " + flight_path + " for map " + mapName + " written to " + csv_file_path)


def get_best_drop_location(mapName, flight_path):
    '''

    :param mapName: string
    :param flight_path: string
    :return: tuple(:x: int, :y: int)

    given a map name and flight direction returns an optimal drop location
    '''

    file_path = './drop_locations'

    if mapName == 'Savage_Main':
        file_path = file_path + '/' + 'drop_' + mapName + '.csv'
    else:
        file_path = file_path + '/' + 'drop_' + mapName + '_' + flight_path + '.csv'

    df = pd.read_csv(file_path)
    index = random.randint(0, len(df) - 1)

    optimal_location = df.iloc[index]
    x = optimal_location['x_location']
    y = optimal_location['y_location']


    return (x, y)




if __name__ == "__main__":
    data_dir = "./data/"
    match_files = []
    telemetry_files = []

    downloader.setup_logging(show_debug=False)
    logging.info("Scanning for match and telemetry files in %s to parse", data_dir)
    for file in os.listdir(data_dir):
        if "_match" in file:
            logging.debug("Match file %s found, adding as match", file)
            match_files.append(file)
        elif "_telemetry" in file:
            logging.debug("Telemetry file %s found, adding as match", file)
            telemetry_files.append(file)




    # this trains the models for predictiing drop locations
    drop_data = jsonparser.get_drop_data(data_dir)
    get_drop_data_by_map(drop_data)
    train_models_drop_locations(drop_data, 20)


    # Just a test telemetry object
    # t = jsonparser.load_pickle(data_dir + telemetry_files[0])
    # zone_info = jsonparser.getZoneStates(t)
    # blue_zones = zone_info[["safetyZonePosition_x", "safetyZonePosition_y", "safetyZoneRadius"]]
    # blue_zones.columns = blue_zones.columns.map({"safetyZonePosition_x": "x",
    #                                              "safetyZonePosition_y": "y",
    #                                              "safetyZoneRadius": "radius"})
    #
    # #data = get_map_data(telemetry_files[:5])
    # #models = train_models(data)
    # #
    # # Get map name, flight path, and location info from telemetry
    # map_n = jsonparser.get_map(t)
    # fp = jsonparser.get_flight_cat_from_telemetry(t)
    # drop = player_info.get_player_paths(t)
    # total = player_info.join_player_and_zone(drop, zone_info)
    #
    # # Load the model (NOTE, must have pickled models that are fit to the data already)
    # model = jsonparser.load_pickle("./models/Savage_Main_nn-model.pickle")
    #
    # # Get a random location to use as the drop location
    # total.dropna(inplace=True)
    # rand_pos = total.sample(n=1)
    # x_ = rand_pos['x'].values[0].item()
    # y_ = rand_pos['y'].values[0].item()
    # print(x_, y_)
    # #
    # # Generate a path (DataFrame)
    # path = gen_path(int(x_), int(y_), blue_zones, 8.5, model)
    # print(path)
    #
    # # Display the path
    # data_visualization.display_player_path(pd.DataFrame(path), None, map_n)

    # """
