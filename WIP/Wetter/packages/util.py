import os
import time
import datetime
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import networkx as nx
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from geopy.distance import geodesic
from matplotlib import pyplot as plt, animation
import matplotlib
from tqdm import tqdm
import glob


def preprocess_data(dpi=96):
    # Step 1: Create a local folder to save the downloads
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    # Step 2: Set the URL of the directory containing .zip files
    base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/"

    # Step 3: Scrape .zip file names
    print("Fetching file list...")
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # Essential files to keep
    keep_list = ["df_all.txt", "dist_mat.txt", "KL_Tageswerte_Beschreibung_Stationen.txt"]
    # Get absolute paths of items to keep
    keep_paths = [os.path.join(data_folder, item) for item in keep_list]
    miss_paths = [path for path in keep_paths if not os.path.exists(path)]

    def delete_file():
        # Delete unnecessary files
        print(f"The following files are missing: {miss_paths}")
        print("Deleting all other files...")
        for file in os.listdir(data_folder):
            if file not in keep_list:
                os.remove(os.path.join(data_folder, file))
        print("All other files deleted.")

    if not miss_paths:
        delete_file()

        # Read data
        local_path = os.path.join(data_folder, "df_all.txt")
        df_all = pd.read_csv(local_path, index_col=0)

        # Download the .txt file
        txt_files = [link.get('href') for link in soup.find_all('a') if link.get('href', '').endswith('.txt')][0]
        local_path = os.path.join(data_folder, txt_files)

        # Read column names from the first line
        with open(local_path, 'r', encoding='latin1') as f:
            columns = f.readline().strip().split()

        # Read the fixed-width formatted file
        df_stat = pd.read_fwf(
            local_path,
            skiprows=2,  # Skip the header and the line of dashes
            header=None,  # no header in the actual data rows
            names=columns,  # set your custom column names
            encoding='latin1'  # To handle special characters
        )
        # Remove rows from df_stat corresponding to all stations which are not present in df_all
        df_stat = df_stat[df_stat['Stations_id'].isin(df_all.columns.values.astype(np.int64))]

        n = len(df_stat)
        local_path = os.path.join(data_folder, 'dist_mat.txt')
        dist_mat = pd.read_csv(local_path).iloc[:, 1:].values
    else:
        zip_files = [link.get('href') for link in soup.find_all('a') if link.get('href', '').endswith('.zip')]
        print(f"Found {len(zip_files)} ZIP files.")

        def download_file(fname):
            file_url = base_url + fname
            local_path = os.path.join(data_folder, fname)
            if os.path.exists(local_path):
                return
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)

        # Step 4: Download ZIP files in parallel
        with ThreadPoolExecutor(max_workers=2 * os.cpu_count()) as executor:
            futures = [executor.submit(download_file, fname) for fname in zip_files]
            for _ in tqdm(as_completed(futures), total=len(futures), ncols=100, desc="Downloading files"):
                pass  # tqdm will update progress as each future completes

        # Step 5: Unzip and extract only files starting with 'produkt'
        print("Unzipping only 'produkt' files...")
        for fname in tqdm(zip_files, ncols=100, desc="Unzipping files"):
            local_zip_path = os.path.join(data_folder, fname)
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                # Filter members whose filenames start with 'produkt'
                member = [m for m in zip_ref.namelist() if os.path.basename(m).startswith('produkt')][0]
                # Optional: Skip if already unzipped
                if os.path.exists(os.path.join(data_folder, member)):
                    continue
                if member:  # Only extract if there are such files
                    zip_ref.extract(member, data_folder)
        print("Selected files extracted.")

        # Read temperature data from all produkt files and compile them into a single dataframe
        local_path = os.path.join(data_folder, "df_all.txt")
        if not os.path.exists(local_path):
            # Find all txt files that start with 'produkt'
            txt_files = glob.glob(os.path.join(data_folder, 'produkt*.txt'))
            # Keys correspond to the station
            keys = [pd.read_csv(f, sep=';', encoding='latin1').loc[:, ["STATIONS_ID"]].values[0, 0] for f in
                    tqdm(txt_files, ncols=100, desc="Reading keys")]
            # Get indices for sorting keys
            idx = sorted(range(len(keys)), key=lambda k: keys[k])
            # Sort keys in ascending order
            keys = [keys[i] for i in idx]
            # Values correspond to the dataframe
            values = [pd.read_csv(f, sep=';', encoding='latin1').loc[:, ["MESS_DATUM", " TMK"]] for f in
                      tqdm(txt_files, ncols=100, desc="Reading values")]
            # Sort values in corresponding order
            values = [values[i] for i in idx]

            # Read and concatenate all data into a single DataFrame
            dates = sorted(set(np.concatenate([f.loc[:, ["MESS_DATUM"]].values.ravel() for f in values])))
            # Initialize dataframe with NaN
            df_all = pd.DataFrame(np.nan, index=dates, columns=keys)
            df_all.index.name = "Date"
            for key, value in zip(keys, values):
                df_all.loc[value.loc[:, ["MESS_DATUM"]].values.ravel(), [key]] = value.loc[:, [" TMK"]].values
            # Replace all -999.0 values with NaN
            df_all.replace(-999.0, np.nan, inplace=True)
            df_all.to_csv(local_path, index=True)
            del txt_files, keys, values, idx, dates, df_all
        df_all = pd.read_csv(local_path, index_col=0)

        # Download the .txt file
        txt_files = [link.get('href') for link in soup.find_all('a') if link.get('href', '').endswith('.txt')][0]
        file_url = base_url + txt_files
        local_path = os.path.join(data_folder, txt_files)

        # Download the file
        print(f"Downloading {txt_files} from {file_url} ...")
        response = requests.get(file_url)
        response.raise_for_status()  # check for errors
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"{txt_files} has been downloaded to {local_path}")

        # Read column names from the first line
        with open(local_path, 'r', encoding='latin1') as f:
            columns = f.readline().strip().split()

        # Read the fixed-width formatted file
        df_stat = pd.read_fwf(
            local_path,
            skiprows=2,  # Skip the header and the line of dashes
            header=None,  # no header in the actual data rows
            names=columns,  # set your custom column names
            encoding='latin1'  # To handle special characters
        )
        # Remove rows from df_stat corresponding to all stations which are not present in df_all
        df_stat = df_stat[df_stat['Stations_id'].isin(df_all.columns.values.astype(np.int64))]

        # Compute matrix of geodesic distances between stations
        n = len(df_stat)
        local_path = os.path.join(data_folder, 'dist_mat.txt')
        if not os.path.exists(local_path):
            # Prepare the positions as (lat, lon) tuples
            pos = list(zip(df_stat['geoBreite'], df_stat['geoLaenge']))

            # Initialize the distance matrix
            dist_mat = np.zeros((n, n))

            # Prepare all unique (i, j) pairs in the upper triangle
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

            def compute_distance(pair):
                i, j = pair
                return (i, j, geodesic(pos[i], pos[j]).kilometers)

            with ThreadPoolExecutor(max_workers=2 * os.cpu_count()) as executor:
                futures = [executor.submit(compute_distance, pair) for pair in
                           tqdm(pairs, ncols=100, desc="Preparing distances")]
                for f in tqdm(as_completed(futures), total=len(futures), ncols=100, desc="Calculating distances"):
                    i, j, dist = f.result()
                    dist_mat[i, j] = dist
                    dist_mat[j, i] = dist  # symmetric

            # Create a DataFrame with station IDs as labels
            dist_mat = pd.DataFrame(dist_mat, index=df_stat['Stations_id'], columns=df_stat['Stations_id'])
            dist_mat.to_csv(local_path)
        dist_mat = pd.read_csv(local_path).iloc[:, 1:].values

        delete_file()

    # Initialize adjacency matrix
    adj_mat = np.zeros((n, n), dtype=np.uint8)
    # TODO Set soft threshold instead of hard
    # Threshold in kilometres
    lim = dist_mat.max() / 10
    adj_mat[(dist_mat > 0) & (dist_mat <= lim)] = 1
    adj_mat[dist_mat > lim] = 0
    # Check if the graph is connected
    if (~adj_mat.any(axis=0)).any():
        raise Exception("Graph is not connected. Choose a higher threshold.")
    adj_mat = pd.DataFrame(adj_mat, index=df_stat['Stations_id'], columns=df_stat['Stations_id'])

    if not os.path.exists("stations.png"):
        # Create a graph
        G = nx.Graph()
        for _, row in tqdm(df_stat.iterrows(), ncols=100, desc="Adding nodes"):
            G.add_node(row['Stations_id'], pos=(row['geoLaenge'], row['geoBreite']))

        # Add edges based on the adjacency matrix
        for i in tqdm(range(n), ncols=100, desc="Adding edges"):
            for j in range(i + 1, n):  # To avoid double-adding and self-loops
                if adj_mat.iloc[i, j] == 1:
                    node_i = adj_mat.index[i]
                    node_j = adj_mat.columns[j]
                    G.add_edge(node_i, node_j)

        # Extract node positions for plotting
        pos = nx.get_node_attributes(G, 'pos')

        # Set up the Cartopy map
        fig, ax = plt.subplots(figsize=(7, 9), dpi=dpi, subplot_kw={'projection': ccrs.Mercator()})
        ax.set_extent([5.5, 15.5, 47, 55], crs=ccrs.PlateCarree())

        # Add country borders
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=1)

        # Draw Germany outline
        countries_shp = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
        for record in shpreader.Reader(countries_shp).records():
            if record.attributes['NAME'] == 'Germany':
                ax.add_geometries(
                    [record.geometry],
                    crs=ccrs.PlateCarree(),
                    edgecolor='red',
                    facecolor='none',
                    linewidth=2
                )

        # Plot edges in black
        for node_i, node_j in tqdm(G.edges, ncols=100, desc="Plotting edges"):
            lon_i, lat_i = pos[node_i]
            lon_j, lat_j = pos[node_j]
            ax.plot([lon_i, lon_j], [lat_i, lat_j], color="black", linewidth=0.7, transform=ccrs.PlateCarree())

        # Draw nodes as points and label them
        for node, (lon, lat) in tqdm(pos.items(), ncols=100, desc="Plotting nodes"):
            ax.plot(lon, lat, 'bo', markersize=4, transform=ccrs.PlateCarree())

        ax.set_title('German Weather Stations')
        plt.tight_layout()
        plt.savefig("stations.png")
        plt.show()

    # Create and save animation, with each node's colour corresponding to the temperature
    if not os.path.exists("stations.gif"):
        # Adjust slicing. Should not exceed 10 years
        df_all_ = df_all[-365:]
        # Extract min and max temperature for heatmap
        min_temp, max_temp = np.nanmin(df_all_.values), np.nanmax(df_all_.values)

        def animate(idx):
            ax.cla()  # Clear the current axes

            ax.set_extent([5.5, 15.5, 47, 55], crs=ccrs.PlateCarree())
            # Add country borders
            ax.add_feature(cfeature.BORDERS, linewidth=1)
            ax.add_feature(cfeature.COASTLINE, linewidth=1)

            # Draw Germany outline
            countries_shp = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
            for record in shpreader.Reader(countries_shp).records():
                if record.attributes['NAME'] == 'Germany':
                    ax.add_geometries(
                        [record.geometry],
                        crs=ccrs.PlateCarree(),
                        edgecolor='red',
                        facecolor='none',
                        linewidth=2
                    )

            # Create a graph
            G = nx.Graph()
            # Extract dataframe corresponding to the relevant nodes at the current timestamp
            df_stat_ = df_stat[df_stat['Stations_id'].isin(df_all_.loc[idx].dropna().index.astype(np.int64))]
            for row, temp in zip(df_stat_.iterrows(), list(df_all_.loc[idx].dropna())):
                # Remove row index
                row = row[1]
                G.add_node(row['Stations_id'], pos=(row['geoLaenge'], row['geoBreite']), temp=temp)

            # Extract adjacency matrix corresponding to relevant nodes at the current timestamp
            adj_mat_ = adj_mat[df_all_.loc[idx].dropna().index.astype(np.int64)].loc[
                df_all_.loc[idx].dropna().index.astype(np.int64)]
            n = len(adj_mat_)
            # Add edges based on the adjacency matrix
            edges = [
                (adj_mat_.index[i], adj_mat_.columns[j])
                for i in range(n) for j in range(i + 1, n) if adj_mat_.iloc[i, j] == 1
            ]
            G.add_edges_from(edges)

            # Extract node positions for plotting
            pos = nx.get_node_attributes(G, 'pos')
            # Plot edges in black
            for node_i, node_j in G.edges:
                lon_i, lat_i = pos[node_i]
                lon_j, lat_j = pos[node_j]
                ax.plot([lon_i, lon_j], [lat_i, lat_j], color="black", linewidth=0.7, transform=ccrs.PlateCarree())

            # Plot nodes as a scatter plot using ax
            ax.scatter(
                [G.nodes[n]['pos'][0] for n in G.nodes],  # Longitude
                [G.nodes[n]['pos'][1] for n in G.nodes],  # Latitude
                c=[G.nodes[n]['temp'] for n in G.nodes],  # Temperature
                cmap=cmap,
                norm=norm,
                s=20,  # Node size, adjust as needed
                edgecolor='k',
                transform=ccrs.PlateCarree(),  # If using Cartopy
                zorder=3
            )

            def get_ordinal(n):
                # Determine the ordinal suffix for a given number
                if n % 10 == 1:
                    return f"{n}st"
                elif n % 10 == 2:
                    return f"{n}nd"
                elif n % 10 == 3:
                    return f"{n}rd"
                else:
                    return f"{n}th"

            def format_int_date(date_int):
                # Parse integer to date
                date = datetime.strptime(str(date_int), "%Y%m%d")
                day = get_ordinal(date.day)
                month = date.strftime("%b")
                year = date.year
                return f"{day} {month} {year}"

            ax.set_title(format_int_date(idx))
            fig.tight_layout()

        def progress_callback(current_frame, total_frames):
            # tqdm uses 0-based, so add 1 to avoid off-by-one errors
            pbar.n = current_frame + 1
            pbar.refresh()

        # Set up the Cartopy map
        fig, ax = plt.subplots(figsize=(7, 9), dpi=dpi, subplot_kw={'projection': ccrs.Mercator()})
        # Set up normalization and colormap before the animation
        norm = matplotlib.colors.Normalize(vmin=min_temp, vmax=max_temp)
        cmap = plt.cm.viridis

        # Create a dummy scatter so the colorbar has something to attach to
        dummy_sc = ax.scatter([], [], c=[], cmap=cmap, norm=norm, s=20)
        fig.colorbar(dummy_sc, ax=ax, orientation='horizontal', shrink=0.5, fraction=0.03, pad=0.005,
                     label='Temperature')
        del dummy_sc  # Remove the dummy scatter after creating the colorbar
        anim = animation.FuncAnimation(fig, animate, frames=list(df_all_.index), interval=125, repeat=True)
        # Create tqdm progress bar with the total number of frames
        pbar = tqdm(total=len(df_all_.index), desc="Frames", ncols=100)
        # Pass the progress_callback to save; tqdm will close itself after
        anim.save("stations.gif", writer="pillow", progress_callback=progress_callback)
        pbar.close()

    # Preprocessing: Ensure that each consecutive snapshot has the exact same active nodes
    # Extract the most recent snapshots
    # TODO Set truncation as variable
    df_all_ = df_all[10 * -365:]
    # Get active stations at every timestamp
    act_nodes = []
    for idx, row in df_all_.iterrows():
        act_nodes.append(row.dropna().index.astype(np.int64))
    # Get stations which are always active at every time instance
    act_nodes = reduce(lambda x, y: x.intersection(y), act_nodes)
    # Extract adjacency matrix corresponding to active nodes
    adj_mat_ = adj_mat[act_nodes].loc[act_nodes]
    # Check if the graph is connected
    if (~adj_mat_.any(axis=0)).any():
        raise Exception("Graph is not connected. Choose a higher threshold.")
    if adj_mat_.empty:
        raise Exception("No nodes present.")