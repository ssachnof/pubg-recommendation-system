import logging

import jsonparser as jsp
import matplotlib.pyplot as plt
from imageio import imread


# Plot the drop locations of each player (in blue), with opacity in relation to their rank in that match
# (more opaque = lower rank), along with the location the first person left the plane (in green)
# and the last person to leave the plane (in red)
def display_drop_locations_by_rank(telemetry, fig, fig_x, fig_y, fig_num, match_num):
    landings = jsp.get_all_landings(telemetry)
    rankings = jsp.get_rankings(telemetry)
    map_name = jsp.get_map(telemetry)

    # Set up plot scale
    if map_name == "Savage_Main":                        # 4km map
        x_max = jsp.MAX_MAP_SIZE * (1/2)
        y_max = x_max
        map_img = imread("savage.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 4.0, 0.0, 4.0])
    elif map_name == "Erangel_Main":                     # 8km map
        x_max = jsp.MAX_MAP_SIZE
        y_max = jsp.MAX_MAP_SIZE
        map_img = imread("erangel.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 8.0, 0.0, 8.0])
    elif map_name ==  "Desert_Main":                     # 8km map
        x_max = jsp.MAX_MAP_SIZE
        y_max = x_max
        map_img = imread("miramar.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 8.0, 0.0, 8.0])
    elif map_name == "DihorOtok_Main":                   # 6km map
        x_max = jsp.MAX_MAP_SIZE * (3/4)
        y_max = x_max
        map_img = imread("vikendi.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 6.0, 0.0, 6.0])

    first_launch, last_launch = jsp.get_flight_data(telemetry)

    if first_launch is not None:
        launch_x = [first_launch['x'], last_launch['x']]
        launch_y = [first_launch['y'], last_launch['y']]

        ax = fig.add_subplot(fig_x, fig_y, fig_num)

        # plot first and last jump locations
        ax.scatter(launch_x[0] / jsp.CM_TO_KM, launch_y[0] / jsp.CM_TO_KM, s=100,
                   color='green', edgecolors='black', zorder=1)
        ax.scatter(launch_x[1] / jsp.CM_TO_KM, launch_y[1] / jsp.CM_TO_KM, s=100,
                   color='red', edgecolors='black', zorder=1)

        # plot line between them
        ax.plot([x_ / jsp.CM_TO_KM for x_ in launch_x],
                [y_ / jsp.CM_TO_KM for y_ in launch_y], 'grey', linestyle='--', marker='', zorder=1)

        # plot each player according to their ranking
        for ranking in rankings:
            landing_loc = landings[ranking['name']]
            # print("Player {} landing at position\t ({}, {}) and ended up rank : {}".format(ranking['name'],
            #                                                                           landing_loc[0],
            #                                                                          landing_loc[1],
            #                                                                          ranking['ranking']))
            if ranking['ranking'] == 1:
                ax.scatter(landing_loc[0] / jsp.CM_TO_KM, landing_loc[1] / jsp.CM_TO_KM,
                           color='yellow', edgecolors='black', zorder=1)
            else:
                ax.scatter(landing_loc[0] / jsp.CM_TO_KM, landing_loc[1] / jsp.CM_TO_KM,
                           color='blue', alpha=1/ranking['ranking'], zorder=1)
        plt.ylim(0, y_max)
        plt.xlim(0, x_max)
        plt.xlabel('km')
        plt.ylabel('km')
        plt.title(map_name)
        plt.savefig('./match_landings/match_{}.png'.format(match_num))
        plt.show()
    else:
        logging.error("Could not get launch data")

# TODO for drop location classif
def display_drop_locations_by_prediction(player_data):
    map_name = player_data.iloc[0]['map']

    # Set up plot scale
    if map_name == "Savage_Main":  # 4km map
        x_max = jsp.MAX_MAP_SIZE * (1 / 2)
        y_max = x_max
        map_img = imread("savage.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 4.0, 0.0, 4.0])
    elif map_name == "Erangel_Main":  # 8km map
        x_max = jsp.MAX_MAP_SIZE
        y_max = jsp.MAX_MAP_SIZE
        map_img = imread("erangel.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 8.0, 0.0, 8.0])
    elif map_name == "Desert_Main":  # 8km map
        x_max = jsp.MAX_MAP_SIZE
        y_max = x_max
        map_img = imread("miramar.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 8.0, 0.0, 8.0])
    elif map_name == "DihorOtok_Main":  # 6km map
        x_max = jsp.MAX_MAP_SIZE * (3 / 4)
        y_max = x_max
        map_img = imread("vikendi.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 6.0, 0.0, 6.0])

def make_circle(safe):
    x_ = safe["safe_x"] / jsp.CM_TO_KM
    y_ = safe["safe_y"] / jsp.CM_TO_KM
    r_ = safe["safe_r"] / jsp.CM_TO_KM
    return plt.Circle((x_, y_), r_, color="midnightblue", alpha=0.75, fill=False)


def display_map_grid(map_name):
    if map_name == "Savage_Main":  # 4km map
        x_max = jsp.MAX_MAP_SIZE * (1 / 2)
        y_max = x_max
        map_img = imread("savage.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 4.0, 0.0, 4.0])
    elif map_name == "Erangel_Main":  # 8km map
        x_max = jsp.MAX_MAP_SIZE
        y_max = jsp.MAX_MAP_SIZE
        map_img = imread("erangel2.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 8.0, 0.0, 8.0])
    elif map_name == "Desert_Main":  # 8km map
        x_max = jsp.MAX_MAP_SIZE
        y_max = x_max
        map_img = imread("miramar2.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 8.0, 0.0, 8.0])
    elif map_name == "DihorOtok_Main":  # 6km map
        x_max = jsp.MAX_MAP_SIZE * (3 / 4)
        y_max = x_max
        map_img = imread("vikendi.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 6.0, 0.0, 6.0])

    plt.ylim(0, y_max)
    plt.xlim(0, x_max)
    plt.xlabel('km')
    plt.ylabel('km')
    plt.minorticks_on()
    plt.grid(b=True, which='major', c='grey', alpha=0.5, linestyle='-')
    plt.grid(b=True, which='minor', c='grey', alpha=0.5, linestyle='-')
    plt.title(map_name)

    plt.show()



def display_player_path(paths_df, map_name):
    #TODO Blue zone display

    if map_name == "Savage_Main":  # 4km map
        x_max = jsp.MAX_MAP_SIZE * (1 / 2)
        y_max = x_max
        map_img = imread("savage.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 4.0, 0.0, 4.0])
    elif map_name == "Erangel_Main":  # 8km map
        x_max = jsp.MAX_MAP_SIZE
        y_max = jsp.MAX_MAP_SIZE
        map_img = imread("erangel2.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 8.0, 0.0, 8.0])
    elif map_name == "Desert_Main":  # 8km map
        x_max = jsp.MAX_MAP_SIZE
        y_max = x_max
        map_img = imread("miramar.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 8.0, 0.0, 8.0])
    elif map_name == "DihorOtok_Main":  # 6km map
        x_max = jsp.MAX_MAP_SIZE * (3 / 4)
        y_max = x_max
        map_img = imread("vikendi.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 6.0, 0.0, 6.0])

    paths_after_drop = paths_df[paths_df.index >= 0.5]

    drop_x = paths_after_drop.iat[0, paths_after_drop.columns.get_loc("x")]
    drop_y = paths_after_drop.iat[0, paths_after_drop.columns.get_loc("y")]
    end_x = paths_after_drop.iat[paths_after_drop.index[-1]-1, paths_after_drop.columns.get_loc("x")]
    end_y = paths_after_drop.iat[paths_after_drop.index[-1]-1, paths_after_drop.columns.get_loc("y")]

    circles = paths_df[["safe_x", "safe_y", "safe_r"]].apply(make_circle, axis=1)

    ax = plt.gca()

    for circle in circles:
        ax.add_artist(circle)

    plt.plot(paths_after_drop['x'] / jsp.CM_TO_KM, paths_after_drop['y'] / jsp.CM_TO_KM, color="black", alpha=0.75)
    plt.plot(drop_x / jsp.CM_TO_KM, drop_y / jsp.CM_TO_KM, 'o', color="green", zorder=1)
    plt.plot(end_x / jsp.CM_TO_KM, end_y / jsp.CM_TO_KM, 'x', color="red", zorder=1)

    plt.ylim(0, y_max)
    plt.xlim(0, x_max)
    plt.xlabel('km')
    plt.ylabel('km')
    plt.minorticks_on()
    plt.grid(b=True, which='major', c='grey', alpha=0.5, linestyle='-')
    plt.grid(b=True, which='minor', c='grey', alpha=0.5, linestyle='-')
    plt.title(map_name)

    plt.show()

if __name__ == "__main__":
    for map in ["Savage_Main", "Erangel_Main", "Desert_Main", "DihorOtok_Main"]:
        display_map_grid(map)

