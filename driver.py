import sys
import os
import random

import pandas as pd

import recommender
import jsonparser
import data_visualization

def printHelp():
    print('usage: python driver.py -l <map_name> [-h] [-p <inputfile>] [-d]\n')

    print('Options')
    print('\t-h : help')
    print('\t-p : Flight path to use. Enter flight path as one of the cardinal directions.')
    print('\t-m : Map Name')

def predict(model, input_data):
    return model.predict(input_data)


def get_drop_model_file_path(map_name, flight_path):
    '''
    :param map_name: string
    :param flight_path: string-> flight path of the plane
    :returns: the path of the trained model given a map name and a flight path
    '''

    if map_name != 'Savage_Main':
        return "./drop_models/model_" + map_name + "_" + flight_path + ".pkl"
    return "./drop_models/model_Savage_Main.pkl"



def main():
    argv = sys.argv[1:]

    path = False
    drop = False
    flight_path = None
    dropAll = True

    if '-h' in argv:
        printHelp()
        exit()

    if '-m' not in argv or "-p" not in argv:
        print ('error: need map name and flight path')
        exit()

    map_name = argv[argv.index('-m') + 1]
    flight_path = argv[argv.index('-p') + 1]

    maps = ["Savage_Main", "DihorOtok_Main", "Erangel_Main", "Desert_Main"]
    flights = ["nn", "ne", "ee", "se", "ss", "sw", "ww", "nw"]

    if map_name not in maps:
        print("Map name not recognized. Please use one of the following:\n")
        for map in maps:
            print(map, end=" ")
    if flight_path not in flights:
        print("Flight path not recognized. Please use one of the following:\n")
        for flight in flights:
            print(flight, end=" ")

    data_dir = "./data/"
    telemetry_files = []

    rand_ind = random.randint(0, len(os.listdir(data_dir)))
    for file in os.listdir(data_dir)[rand_ind:rand_ind+100]:
        if "_telemetry" in file:
            t = jsonparser.load_pickle(data_dir + file)
            map_n = jsonparser.get_map(t)
            if map_n == map_name:
                telemetry_files.append(file)

    model = jsonparser.load_pickle("./path_models/{}-model.pickle".format(map_name))

    # Get random telemetry file to use as initial blue zone
    t = jsonparser.load_pickle(data_dir + telemetry_files[random.randint(0, len(telemetry_files)-1)])
    zone_info = jsonparser.get_zone_states(t)
    blue_zones = (zone_info[zone_info.isGame == 1.0]
        [["safetyZonePosition_x", "safetyZonePosition_y", "safetyZoneRadius"]])
    blue_zones.columns = blue_zones.columns.map({"safetyZonePosition_x": "x",
                                                 "safetyZonePosition_y": "y",
                                                 "safetyZoneRadius": "radius"})

    drop_x, drop_y = recommender.get_best_drop_location(map_name, flight_path)
    path = recommender.gen_path(int(drop_x), int(drop_y), blue_zones, 7.5, model)
    data_visualization.display_player_path(pd.DataFrame(path), map_name)

    return 0


if __name__ == '__main__':
    main()
