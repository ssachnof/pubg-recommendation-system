import pandas as pd
from pandas.io.json import json_normalize
import requests
import gzip
import shutil
import os
import time
import logging
import match
import json
import sys
import multiprocessing
import pickle
from threading import Thread
from pathlib import Path
import argparse


class APIKey:
    def __init__(self, key_string):
        self.key_string = key_string
        self.limit_max = -1
        self.limit_remaining = -1
        self.limit_reset = -1


class MatchSamplesResponse:
    def __init__(self, response_code):
        self.response_code = response_code
        self.match_ids = []


def main():
    parser = argparse.ArgumentParser("Downloads PUBG match data")
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--downloaddir')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--apikeys')
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()
    api_keys_filename = "api_keys.txt"
    if args.apikeys is not None:
        api_keys_filename = args.apikeys
    download_directory = "data/"
    if args.downloaddir is not None:
        download_directory = args.downloaddir
    download_threads = os.cpu_count()  # Create threads equivalent to number of CPU cores
    if args.threads is not None:
        download_threads = int(args.threads)
    setup_logging(args.debug)
    logging.info("Using %i threads to download PUBG data", download_threads)

    # Get API keys
    logging.debug("Getting API keys from file %s", api_keys_filename)
    api_keys = []
    with open(api_keys_filename) as api_keys_file:
        for key in api_keys_file:
            api_keys.append(APIKey(key.rstrip()))

    while True:
        # Get matches and test keys
        logging.debug("Cycling through API keys to retrieve sample set of match ids")
        sample_matches = None
        match_id_queue = []
        for key in api_keys:
            if float(key.limit_reset) < time.time():
                key.limit_remaining = key.limit_max
            if key.limit_remaining == 0:
                logging.info("API key ending in %s has no uses remaining, skipping and trying next key in pool",
                             key[-4::])
                continue
            sample_matches = get_sample_matches(key)
            if sample_matches.response_code == 200:
                match_id_queue.extend(sample_matches.match_ids)
                logging.debug("Sample match ids acquired")
                break
        if len(match_id_queue) < 1:
            logging.error("Could not get any match ids with any keys in pool")

        # Query each match on the API and dump its info
        logging.debug("Retrieving info about each match")
        if sample_matches is None:
            logging.exception("No match ids found to download, quitting")
            exit()
        telemetry_queue = multiprocessing.Queue()
        download_count = 0
        for i in range(download_threads):
            new_thread = Thread(target=handle_telemetry, args=(telemetry_queue, download_directory, download_count))
            new_thread.daemon = True
            new_thread.start()
        for count, current_match in enumerate(match_id_queue):
            logging.debug("Retrieving stats for match %i of %i", count, len(match_id_queue))
            match_object = get_match_stats(current_match)
            telemetry_queue.put(match_object)
        logging.info("Downloaded %i new matches from given sample list", download_count)
        if (not args.loop) or download_count < 1:
            logging.info("Downloader finished")
            break


def handle_telemetry(telemetry_queue, download_directory, download_count):
    logging.debug("Downloader thread active")
    while True:
        match_object = telemetry_queue.get()
        # patch_version = match_object.patch_version
        match_date = match_object.start_time[:10]
        map_name = match_object.map

        # Save telemetry if it's new
        # match_file_name = "data/" + match_date + "/" + map_name + "/" + match_object.match_id + "_match.pickle"
        match_file_name = download_directory + match_object.match_id + "_match.pickle"
        match_data_file = Path(match_file_name)
        if match_data_file.is_file():
            logging.info("Match id %s already saved to disk, skipping download", match_object.match_id)
        else:
            logging.debug("Worker process getting telemetry for match id %s", match_object.match_id)
            parsed_telemetry = get_telemetry(match_object.telemetry_url)
            download_count += 1
            # telemetry_file_name = download_directory + match_date + "/" + map_name + "/" + match_object.match_id\
            #                       + "_telemetry.pickle"
            telemetry_file_name = download_directory + match_object.match_id + "_telemetry.pickle"
            write_pickle(match_file_name, match_object)
            write_pickle(telemetry_file_name, parsed_telemetry)
            logging.info("Saved match id %s", match_object.match_id)


def setup_logging(show_debug):
    root = logging.getLogger()
    if show_debug:
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)


# Writes any object into a pickle file so it can be loaded later
def write_pickle(file_path, object_to_pickle):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError:
            logging.error("Could not create directory to store pickle at path %s", file_path)
    with open(file_path, 'wb') as output_file:  # Write to file, truncating (not appending), in binary output mode
        logging.debug("Writing object into file %s", file_path)
        pickle.dump(object_to_pickle, output_file, pickle.HIGHEST_PROTOCOL)


# Get a list of random match IDs from the PUBG API
def get_sample_matches(api_key):
    logging.info("Getting sample matches")
    logging.debug("Using API key ending in %s to get sample matches", api_key.key_string[-4:])
    # Header for auth
    header = {"Authorization": "Bearer %s" % api_key.key_string,
              "Accept": "application/vnd.api+json"}

    # Get the API call data
    api_result = requests.get("https://api.pubg.com/shards/steam/samples", headers=header)
    api_key.limit_remaining = api_result.headers.get("X-Ratelimit-Remaining")
    api_key.limit_max = api_result.headers.get("X-Ratelimit-Limit")
    api_key.limit_reset = api_result.headers.get("X-Ratelimit-Reset")
    sample_matches = MatchSamplesResponse(api_result.status_code)

    if api_result.status_code == 200:
        logging.debug("Received sample matches (HTTP status code 200), extracting match ids")
        # Dump json data into pandas for better formatting
        samples_df = pd.DataFrame(api_result.json())

        # Extract just the match data
        samples_norm = json_normalize(samples_df.loc['relationships'])
        match_ids_dataframe = samples_norm['matches.data'].apply(pd.Series).T[0].apply(pd.Series)
        sample_matches.match_ids = match_ids_dataframe['id'].values
    else:
        logging.error("HTTP status code %s encountered while retrieving matches", api_result.status_code)

    return sample_matches


# Given a match ID, pull the data for that match and return it as a pandas DataFrame
def get_match_stats(match_id):
    logging.debug("Getting match stats for match id %s", match_id)
    header = {"accept": "application/vnd.api+json"}
    apireq = requests.get("https://api.pubg.com/shards/steam/matches/%s" % match_id, headers=header)

    # API response is structured: { data : {...}
    #                               included: [...] }
    # where data has information about the IDs of the match objects (players, rosters, etc) and
    # included contains links to the actual objects about the match (look up by ID from data)

    logging.debug("Match data request completed, parsing response")
    # Import the "data" portion of the API response into Pandas
    api_response_data_dataframe = pd.DataFrame(apireq.json()['data'])
    # Import the "included" portion of the API response into Pandas
    api_response_included_dataframe = pd.DataFrame(apireq.json()['included'])

    # Get the telemetry URL from its asset UID.
    # The telemetry UID is different from the match UID so we must retrieve the telemetry UID.
    # The telemetry UID the only "id" field in the only "data" entry of "type": "asset" ...
    # ... under "data" -> "relationships" -> "assets"

    # Get the assets portion of the data
    assets = pd.DataFrame(api_response_data_dataframe.loc['assets', 'relationships'])['data'].apply(pd.Series)
    # The telemetry id is the only entry of type "data" under section "assets", so grab it
    telemetry_id = assets[assets.type == "asset"]['id'].values[0]
    # Get the telemetry object from the "included" dataframe
    telemetry_object = api_response_included_dataframe[api_response_included_dataframe.id == telemetry_id]
    # Get the url from the telemetry object
    telemetry_url = telemetry_object['attributes'].apply(pd.Series)['URL'].values[0]

    # Extract match statistics and create a match object
    game_mode = api_response_data_dataframe.loc['gameMode', 'attributes']
    game_map = api_response_data_dataframe.loc['mapName', 'attributes']
    start_time = api_response_data_dataframe.loc['createdAt', 'attributes']
    duration = api_response_data_dataframe.loc['duration', 'attributes']
    # patch_version = api_response_data_dataframe.loc['patchVersion', 'attributes']
    match_data_object = match.match(match_id, game_mode, game_map, start_time, duration, telemetry_url)
    return match_data_object


def get_telemetry(telemetry_url):
    logging.debug("Retrieving telemetry data from url %s", telemetry_url)
    response = requests.get(telemetry_url)
    logging.debug("Parsing telemetry JSON")
    parsed_telemetry = json.loads(response.text)
    return parsed_telemetry


# Given location of gzips and location of where to extract to -> unzips gzip files
def extract_gzip(gzip_indir, gzip_outdir):
    # Check if directories exist
    if not os.path.isdir(gzip_indir):
        print("Cannot find directory '" + gzip_indir + "'")
        exit()
    elif not os.path.isdir(gzip_outdir):
        print("Cannot find directory '" + gzip_outdir + "'")
        exit()

    # Get name of all files to unzip
    files = os.listdir(gzip_indir)
    for f in files:
        outfile = f.replace(".gz", "")
        try:
            with gzip.open(gzip_indir + f, 'rb') as g:
                print("Copying '" + f + "'")

                # Copy files to output directory
                with open(gzip_outdir + outfile, 'wb') as g_copy:
                    shutil.copyfileobj(g, g_copy)
        except IOError:
            print("Unable to copy file '" + gzip_indir + f + "'\n")

    print("Copying data - done")


if __name__ == "__main__":
    main()
