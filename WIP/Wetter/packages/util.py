import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import networkx as nx
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from geopy.distance import geodesic
from matplotlib import pyplot as plt
from tqdm import tqdm
import glob

def download_data():
    # Step 1: Set the URL of the directory containing .zip files
    base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/"

    # Step 2: Create a local folder to save the downloads
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    # Step 3: Scrape .zip file names
    print("Fetching file list...")
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    zip_files = [link.get('href') for link in soup.find_all('a') if link.get('href', '').endswith('.zip')]
    print(f"Found {len(zip_files)} ZIP files.")

    # Step 4: Download each ZIP file
    for fname in tqdm(zip_files, ncols=100, desc="Downloading files"):
        file_url = base_url + fname
        local_path = os.path.join(data_folder, fname)

        # Skip if already downloaded
        if os.path.exists(local_path):
            continue

        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    size = f.write(chunk)
    print("All downloads complete.")

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

    # TODO: Revert
    # # Step 6: Delete all .zip files in the output folder
    # print("Deleting .zip files...")
    # for file in os.listdir(data_folder):
    #     if file.endswith('.zip'):
    #         os.remove(os.path.join(data_folder, file))
    # print(".zip files deleted.")

    # Step 7: Download the .txt file
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

    # TODO There is a discrepancy between the number of stations in the txt file and the number of zip files. Ensure that the correct number is shown using the zip file
    # Read the fixed-width formatted file
    df = pd.read_fwf(
        local_path,
        skiprows=2,  # Skip the header and the line of dashes
        header=None,  # no header in the actual data rows
        names=columns,  # set your custom column names
        encoding='latin1'  # To handle special characters
    )

    # Compute matrix of geodesic distances between stations
    n = len(df)
    local_path = os.path.join(data_folder, 'node.txt')
    if not os.path.exists(local_path):
        # Prepare the positions as (lat, lon) tuples
        pos = list(zip(df['geoBreite'], df['geoLaenge']))

        # Initialize the distance matrix
        dist_mat = np.zeros((n, n))

        # Prepare all unique (i, j) pairs in the upper triangle
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]


        def compute_distance(pair):
            i, j = pair
            return (i, j, geodesic(pos[i], pos[j]).kilometers)


        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            futures = [executor.submit(compute_distance, pair) for pair in
                       tqdm(pairs, ncols=100, desc="Preparing distances")]
            for f in tqdm(as_completed(futures), total=len(futures), ncols=100, desc="Calculating distances"):
                i, j, dist = f.result()
                dist_mat[i, j] = dist
                dist_mat[j, i] = dist  # symmetric

        # Create a DataFrame with station IDs as labels
        dist_mat = pd.DataFrame(dist_mat, index=df['Stations_id'], columns=df['Stations_id'])
        dist_mat.to_csv(local_path)
    dist_mat = pd.read_csv(local_path).iloc[:, 1:].values

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
    adj_mat = pd.DataFrame(adj_mat, index=df['Stations_id'], columns=df['Stations_id'])

    # TODO Toggle and adjust
    plot_static_graph = False
    if plot_static_graph:
        # Create a graph
        G = nx.Graph()
        for _, row in df.iterrows():
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
        fig, ax = plt.subplots(figsize=(7, 9), subplot_kw={'projection': ccrs.Mercator()})
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
        for node1, node2 in tqdm(G.edges, ncols=100, desc="Plotting edges"):
            lon1, lat1 = pos[node1]
            lon2, lat2 = pos[node2]
            ax.plot([lon1, lon2], [lat1, lat2], color="black", linewidth=0.7, transform=ccrs.PlateCarree())

        # Draw nodes as points and label them
        for node, (lon, lat) in tqdm(pos.items(), ncols=100, desc="Plotting nodes"):
            ax.plot(lon, lat, 'bo', markersize=4, transform=ccrs.PlateCarree())

        ax.set_title('German Weather Stations')
        plt.tight_layout()
        plt.savefig("stations.png")
        plt.show()

    # TODO Read temperature data from all produkt files and compile them into a single dataframe
    # Find all txt files that start with 'produkt'
    txt_files = glob.glob(os.path.join(data_folder, 'produkt*.txt'))

    txt_files = txt_files[:10]
    # TODO Create dictinary, with keys corresponding to the station, and values corresponding to the dataframe
    all_dfs = {pd.read_csv(f, sep=';', encoding='latin1').loc[:, ["STATIONS_ID"]].values[0, 0]: pd.read_csv(f, sep=';', encoding='latin1').loc[:,["MESS_DATUM"," TMK"]] for f in tqdm(txt_files, ncols=100, desc="Reading dataframes")}  # set correct encoding if needed
    # Read and concatenate all found txt files into a single DataFrame
    # [f.loc[:, ["MESS_DATUM"]].values for f in all_dfs.values()]
    pass

# TODO Create function
def preprocess_data():
    pass

# TODO Create function
def delete_data():
    pass