import napari
import pandas as pd


filename = '/Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3.nd2'
tracks_filename = '/Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3_Position.csv'

track_positions = pd.read_csv(tracks_filename, header=0, skiprows=(0, 1, 2))
tracks = track_positions.sort_values(['TrackID', 'Time'])
tracks['Time'] -= 1  # Imaris uses 1-indexing
columns = ['TrackID', 'Time'] + [f'Position {i}' for i in 'ZYX']

with napari.gui_qt():
    v = napari.Viewer()
    layers = v.open(filename)
    v.add_tracks(
        tracks[columns].to_numpy(),
        properties=tracks,
        color_by='TrackID',
    )