import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import altair as alt
from clyent import color
from mpmath.libmp import normalize
from sqlalchemy.dialects.mssql.information_schema import columns


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def plot_dimensionality_reduction(X_tr, y_tr, Y_label = None, method='PCA', filename = None, elev=30, azim=45, random_state = 42):
    """
    Plot 2D and 3D projections of the data using the specified dimensionality reduction method.

    Parameters:
    - X_tr: array-like, shape (n_samples, n_features)
    - y_tr: array-like, shape (n_samples,)
    - method: str, one of {'PCA', 't-SNE', 'UMAP'} (default: 'PCA')
    - elev: elevation angle for 3D plot (default: 30)
    - azim: azimuth angle for 3D plot (default: 45)
    """

    # Check the method and perform the appropriate dimensionality reduction
    if method == 'PCA':
        # PCA with 2 components
        pca_2d = PCA(n_components=2, random_state = random_state)
        X_reduced_2d = pca_2d.fit_transform(X_tr)

        # PCA with 3 components
        pca_3d = PCA(n_components=3, random_state = random_state)
        X_reduced_3d = pca_3d.fit_transform(X_tr)

    elif method == 'tSNE':
        # t-SNE with 2 components
        tsne_2d = TSNE(n_components=2, random_state=random_state)
        X_reduced_2d = tsne_2d.fit_transform(X_tr)

        # t-SNE with 3 components
        tsne_3d = TSNE(n_components=3, random_state=random_state)
        X_reduced_3d = tsne_3d.fit_transform(X_tr)

    elif method == 'UMAP':
        # UMAP with 2 components
        umap_2d = umap.UMAP(n_components=2, random_state=random_state)
        X_reduced_2d = umap_2d.fit_transform(X_tr)

        # UMAP with 3 components
        umap_3d = umap.UMAP(n_components=3, random_state=random_state)
        X_reduced_3d = umap_3d.fit_transform(X_tr)

    else:
        raise ValueError("Method must be one of {'PCA', 't-SNE', 'UMAP'}")



    fig = plt.figure(figsize=(14, 8))

    # 2D Plot
    ax1 = fig.add_subplot(121)
    scatter = ax1.scatter(X_reduced_2d[:, 0], X_reduced_2d[:, 1], c=y_tr, cmap='plasma')
    ax1.set_title(f'{method} with 2 Components')
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')

    # Set grid color to gray and transparent background
    ax1.grid(True, color='gray', linestyle='-', linewidth=0.7)
    ax1.set_facecolor('none')  # Transparent background

        # Mapping classes to labels
    if Y_label is not None:
        handles, _ = scatter.legend_elements()
        labels = [Y_label[int(label)] for label in np.unique(y_tr)]  # Map class numbers to labels
        legend = ax1.legend(handles, labels, title="Classes")
        legend.get_frame().set_facecolor('white')  # White legend background

    # 3D Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.view_init(elev=elev, azim=azim)
    scatter = ax2.scatter(X_reduced_3d[:, 0], X_reduced_3d[:, 1], X_reduced_3d[:, 2], c=y_tr, cmap='plasma')
    ax2.set_title(f'{method} with 3 Components')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_zlabel('Component 3')

    # Set grid color to gray and transparent background
    ax2.grid(True, color='gray', linestyle='-', linewidth=0.7)
    ax2.set_facecolor('none')  # Transparent background

    if Y_label is not None:
        handles, _ = scatter.legend_elements()
        labels = [Y_label[int(label)] for label in np.unique(y_tr)]  # Map class numbers to labels
        legend = ax2.legend(handles, labels, title="Classes")
        legend.get_frame().set_facecolor('white')  # White legend background

    # Tight layout and transparent figure background
    plt.tight_layout()
    fig.patch.set_alpha(0)  # Transparent figure background
    if filename is not None:
        plt.savefig(filename, dpi=1200, transparent=True)
    plt.show()


def initial_report(data, target, method='PCA', filename=None, scale=False, elev=30, azim=45):
    data_clean = data.dropna()
    X = data_clean.drop(columns=target)
    Y = data_clean[target]

    if not pd.api.types.is_numeric_dtype(Y):
        categories = data_clean[target].sort_values().unique().tolist()
        data_clean[target + '_cod'] = pd.Categorical(data_clean[target], categories = categories, ordered = True)
        data_clean[target + '_cod'] = data_clean[target + '_cod'].cat.codes

        Y_label = dict(zip(data_clean['Padrao_cod'].sort_values().unique(), data_clean['Padrao'].sort_values().unique()))
        Y = data_clean[target + '_cod']

    if scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X.select_dtypes(include=[np.number]))
    plot_dimensionality_reduction(X, Y, Y_label = Y_label, method=method, filename=filename, elev=elev, azim=azim)


def initial_report(data, target, method='PCA', filename=None, scale=False, elev=30, azim=45):
    data_clean = data.dropna()
    X = data_clean.drop(columns=target)
    Y = data_clean[target]

    if not pd.api.types.is_numeric_dtype(Y):
        categories = data_clean[target].sort_values().unique().tolist()
        data_clean[target + '_cod'] = pd.Categorical(data_clean[target], categories = categories, ordered = True)
        data_clean[target + '_cod'] = data_clean[target + '_cod'].cat.codes

        Y_label = dict(zip(data_clean['Padrao_cod'].sort_values().unique(), data_clean['Padrao'].sort_values().unique()))
        Y = data_clean[target + '_cod']

    if scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X.select_dtypes(include=[np.number]))
    plot_dimensionality_reduction(X, Y, Y_label = Y_label, method=method, filename=filename, elev=elev, azim=azim)


def clean_pde_data(data_raw, return_clean = True):
    data = data_raw.copy()

    data = data[data['COD_NAO_LEITURA'].isna()]

    data['PDE'] = data['PDE'].apply(
        lambda x: x.zfill(10)
        )
    data['COD_PDE'] = data['COD_PDE'].apply(
        lambda x: x.zfill(10)
        )

    data['QTD_CONSUMO'] = data['QTD_CONSUMO'].astype(int)
    data['QTD_CONSUMO_MEDIO'] = data['QTD_CONSUMO_MEDIO'].astype(int)

    data = data[data['STATUS_LEITURA'] != 'ANULADO']

    data['DATA_LEITURA_REGISTRO'] = data['DATA_LEITURA_REGISTRO'].str.replace('02-29', '02-28')
    data['timestamp'] = pd.to_datetime(data['DATA_LEITURA_REGISTRO'], format = '%Y-%m-%d %H:%M:%S.%f')
    data['DATA_INCLUSAO'] = pd.to_datetime(data['DATA_INCLUSAO'])

    data = data.sort_values(by = ['PDE', 'DATA_LEITURA_REGISTRO', 'DATA_INCLUSAO'], ascending = [True, True, False])
    data = data.drop_duplicates(subset = ['PDE', 'DATA_LEITURA_REGISTRO', 'DATA_LEITURA_REGISTRO'], keep = 'first')

    data['YEAR'] = data['timestamp'].dt.year
    data['DATE'] = data['timestamp'].dt.year.astype(str) + '-' + data['timestamp'].dt.month.astype(str)
    # data['timestamp'] = data['timestamp'].dt.date

    # Set the 'timestamp' column as the index
    data.set_index('timestamp', inplace = True)
    data.index = pd.to_datetime(data.index)

    if return_clean: data = data[['PDE', 'QTD_CONSUMO', 'QTD_CONSUMO_MEDIO', 'YEAR', 'DATE']]

    return data


def dual_boxplot(data, var_x, var_y1, var_y2, var_hue, features_map, plot_map, clip_yaxis = None):
    """
    Plots two violin plots side by side with custom options.

    Parameters:
    data : DataFrame
        The data frame containing the data for plotting.
    var_x : str
        The column name to use for the x-axis.
    var_y1 : str
        The first column name to use for the y-axis in the first plot.
    var_y2 : str
        The second column name to use for the y-axis in the second plot.
    var_hue : str
        The column name to use for hue (coloring) in both plots.
    features_map : dict
        Dictionary containing customization options for each variable.
    plot_map : dict
        Dictionary containing plot-wide customization options.
    """
    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize = (14, 6))

    # Customize color palette and order for var_hue if specified
    palette = features_map.get(var_hue, {}).get('palette', 'viridis')
    hue_order = features_map.get(var_hue, {}).get('order', None)

    # Plot the first violin plot with customizations
    sns.boxplot(data = data, x = var_x, y = var_y1, hue = var_hue, ax = axs[0],
                palette = palette, hue_order = hue_order)
    axs[0].set_title(plot_map['title_text'].format(var_x = features_map.get(var_x, {}).get('label', var_x),
                                                   var_hue = features_map.get(var_hue, {}).get('label', var_hue)),
                     fontsize = plot_map.get('title_fontsize', 18))
    axs[0].set_xlabel(features_map.get(var_x, {}).get('label', var_x), fontsize = plot_map.get('label_fontsize', 16))
    axs[0].set_ylabel(features_map.get(var_y1, {}).get('label', var_y1), fontsize = plot_map.get('label_fontsize', 16))
    if clip_yaxis is not None:
        axs[0].set_ylim(clip_yaxis)

    # Plot the second violin plot with customizations
    sns.boxplot(data = data, x = var_x, y = var_y2, hue = var_hue, ax = axs[1],
                palette = palette, hue_order = hue_order)
    axs[1].set_title(plot_map['title_text'].format(var_x = features_map.get(var_x, {}).get('label', var_x),
                                                   var_hue = features_map.get(var_hue, {}).get('label', var_hue)),
                     fontsize = plot_map.get('title_fontsize', 18))
    axs[1].set_xlabel(features_map.get(var_x, {}).get('label', var_x), fontsize = plot_map.get('label_fontsize', 16))
    axs[1].set_ylabel(features_map.get(var_y2, {}).get('label', var_y2), fontsize = plot_map.get('label_fontsize', 16))

    # Set tick label font sizes
    for ax in axs:
        ax.tick_params(axis = 'x', labelsize = plot_map.get('tick_fontsize', 14))
        ax.tick_params(axis = 'y', labelsize = plot_map.get('tick_fontsize', 14))

    # Set legend with custom font size
    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(handles, labels, title = features_map.get(var_hue, {}).get('label', var_hue),
                  title_fontsize = plot_map.get('legend_fontsize', 14),
                  fontsize = plot_map.get('legend_fontsize', 12))

    plt.tight_layout()
    plt.show()


def dual_boxplot_hued(data, var_x, var_y, var_hue1, var_hue2, features_map, plot_map):
    """
    Plots two box plots side by side with custom hue for the same y variable.

    Parameters:
    data : DataFrame
        The data frame containing the data for plotting.
    var_x : str
        The column name to use for the x-axis.
    var_y : str
        The column name to use for the y-axis in both plots.
    var_hue1 : str
        The column name to use for hue (coloring) in the first plot.
    var_hue2 : str
        The column name to use for hue (coloring) in the second plot.
    features_map : dict
        Dictionary containing customization options for each variable.
    plot_map : dict
        Dictionary containing plot-wide customization options.
    """
    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize = (14, 6))

    # Customize color palettes and orders for var_hue1 and var_hue2 if specified
    palette1 = features_map.get(var_hue1, {}).get('palette', 'viridis')
    hue_order1 = features_map.get(var_hue1, {}).get('order', None)

    palette2 = features_map.get(var_hue2, {}).get('palette', 'viridis')
    hue_order2 = features_map.get(var_hue2, {}).get('order', None)

    # Plot the first box plot with customizations for hue1
    sns.boxplot(data = data, x = var_x, y = var_y, hue = var_hue1, ax = axs[0],
                palette = palette1, hue_order = hue_order1)
    axs[0].set_title(plot_map['title_text'].format(var_x = features_map.get(var_x, {}).get('label', var_x),
                                                   var_hue = features_map.get(var_hue1, {}).get('label', var_hue1)),
                     fontsize = plot_map.get('title_fontsize', 18))
    axs[0].set_xlabel(features_map.get(var_x, {}).get('label', var_x), fontsize = plot_map.get('label_fontsize', 16))
    axs[0].set_ylabel(features_map.get(var_y, {}).get('label', var_y), fontsize = plot_map.get('label_fontsize', 16))

    # Plot the second box plot with customizations for hue2
    sns.boxplot(data = data, x = var_x, y = var_y, hue = var_hue2, ax = axs[1],
                palette = palette2, hue_order = hue_order2)
    axs[1].set_title(plot_map['title_text'].format(var_x = features_map.get(var_x, {}).get('label', var_x),
                                                   var_hue = features_map.get(var_hue2, {}).get('label', var_hue2)),
                     fontsize = plot_map.get('title_fontsize', 18))
    axs[1].set_xlabel(features_map.get(var_x, {}).get('label', var_x), fontsize = plot_map.get('label_fontsize', 16))
    axs[1].set_ylabel(features_map.get(var_y, {}).get('label', var_y), fontsize = plot_map.get('label_fontsize', 16))

    # Set tick label font sizes
    for ax in axs:
        ax.tick_params(axis = 'x', labelsize = plot_map.get('tick_fontsize', 14))
        ax.tick_params(axis = 'y', labelsize = plot_map.get('tick_fontsize', 14))

    # Set legends with custom font sizes
    handles1, labels1 = axs[0].get_legend_handles_labels()
    axs[0].legend(handles1, labels1, title = features_map.get(var_hue1, {}).get('label', var_hue1),
                  title_fontsize = plot_map.get('legend_fontsize', 14),
                  fontsize = plot_map.get('legend_fontsize', 12))

    handles2, labels2 = axs[1].get_legend_handles_labels()
    axs[1].legend(handles2, labels2, title = features_map.get(var_hue2, {}).get('label', var_hue2),
                  title_fontsize = plot_map.get('legend_fontsize', 14),
                  fontsize = plot_map.get('legend_fontsize', 12))

    plt.tight_layout()
    plt.show()





def features_hued_analysis(data, var_x, var_y, var_hue, features_map, plot_map):
    """
    Plots a grid of three plots:
    - First row: A single box plot with hue.
    - Second row: Two side-by-side plots, each combining a violin plot and a box plot.

    Parameters:
    data : DataFrame
        The data frame containing the data for plotting.
    var_x : str
        The column name to use for the x-axis.
    var_y : str
        The column name to use for the y-axis in the second row.
    var_hue : str
        The column name to use for hue (coloring) in both plots.
    features_map : dict
        Dictionary containing customization options for each variable.
    plot_map : dict
        Dictionary containing plot-wide customization options.
    """
    # Create a figure with a 2-row, 2-column grid (first row spans both columns)
    fig = plt.figure(figsize = (14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios = [1, 1])

    # First row (wide plot across both columns): Box plot with hue
    ax1 = fig.add_subplot(gs[0, :])
    palette_hue = features_map.get(var_hue, {}).get('palette', 'plasma')
    hue_order_hue = features_map.get(var_hue, {}).get('order', None)

    sns.boxplot(data = data, x = var_x, y = var_y, hue = var_hue, ax = ax1,
                palette = palette_hue, hue_order = hue_order_hue)
    ax1.set_title(plot_map['title_text'].format(var_x = features_map.get(var_x, {}).get('label', var_x),
                                                var_hue = features_map.get(var_hue, {}).get('label', var_hue)),
                  fontsize = plot_map.get('title_fontsize', 18))
    ax1.set_xlabel(features_map.get(var_x, {}).get('label', var_x), fontsize = plot_map.get('label_fontsize', 16))
    ax1.set_ylabel(features_map.get(var_y, {}).get('label', var_y), fontsize = plot_map.get('label_fontsize', 16))

    # Second row, first plot (violin + box plot for var_y)
    ax2 = fig.add_subplot(gs[1, 0])
    palette_y = features_map.get(var_x, {}).get('palette', 'viridis')
    hue_order_y = features_map.get(var_x, {}).get('order', None)

    sns.violinplot(data = data, x = var_x, y = var_y, ax = ax2, palette = palette_y, hue_order = hue_order_y,
                   split = True)
    sns.boxplot(data = data, x = var_x, y = var_y, ax = ax2,
                palette = palette_y, hue_order = hue_order_y, boxprops = {'facecolor': 'none'}, width = 0.3)

    ax2.set_title(plot_map['title_text'].format(var_x = features_map.get(var_y, {}).get('label', var_y),
                                                var_hue = features_map.get(var_x, {}).get('label', var_x)),
                  fontsize = plot_map.get('title_fontsize', 18))
    ax2.set_xlabel(features_map.get(var_x, {}).get('label', var_x), fontsize = plot_map.get('label_fontsize', 16))
    ax2.set_ylabel(features_map.get(var_y, {}).get('label', var_y), fontsize = plot_map.get('label_fontsize', 16))

    # Second row, second plot (violin + box plot for var_hue)
    ax3 = fig.add_subplot(gs[1, 1])
    sns.violinplot(data = data, x = var_hue, y = var_y, ax = ax3, palette = palette_hue, hue_order = hue_order_hue,
                   split = True)
    sns.boxplot(data = data, x = var_hue, y = var_y, ax = ax3,
                palette = palette_hue, hue_order = hue_order_hue, boxprops = {'facecolor': 'none'}, width = 0.3)

    ax3.set_title(plot_map['title_text'].format(var_x = features_map.get(var_y, {}).get('label', var_y),
                                                var_hue = features_map.get(var_hue, {}).get('label', var_hue)),
                  fontsize = plot_map.get('title_fontsize', 18))
    ax3.set_xlabel(features_map.get(var_hue, {}).get('label', var_hue), fontsize = plot_map.get('label_fontsize', 16))
    ax3.set_ylabel(features_map.get(var_y, {}).get('label', var_y), fontsize = plot_map.get('label_fontsize', 16))

    # Set tick label font sizes for all axes
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis = 'x', labelsize = plot_map.get('tick_fontsize', 14))
        ax.tick_params(axis = 'y', labelsize = plot_map.get('tick_fontsize', 14))

    # Set legends with custom font sizes
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, title = features_map.get(var_hue, {}).get('label', var_hue),
               title_fontsize = plot_map.get('legend_fontsize', 14),
               fontsize = plot_map.get('legend_fontsize', 12))

    plt.tight_layout()
    plt.show()




