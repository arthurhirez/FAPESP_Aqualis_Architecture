import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


def create_threshold_analysis(data, feature = 'Total_Unidades', plot_map = None,
                              tags = (True, False), filter_data = False,
                              plot_analsis = True, threshold = None, xlim = None,
                              ):
    """
    Plots the dataset sizes for thresholds and the KDE plot with layered fills.

    Parameters:
    - df: DataFrame containing the data.
    - feature: Column name for analysis (default is 'Total_Unidades').
    - vline_threshold: Threshold for vertical line (default is None, which sets it to max value of feature).
    - plot_map: Dictionary for plot settings.
    - xl im: Tuple for x-axis limits of the KDE plot (default is None, which means automatic limits).
    - color_A: Color for the '<= Threshold' line (default is 'red').
    - color_B: Color for the '>= Threshold' line (default is 'blue').
    """

    if tags[0] == tags[1]:
        raise ValueError('Tags should be like (True, False) or (False, True)')

    color_A = 'red' if tags[0] == False else 'blue'
    color_B = 'blue' if tags[0] == False else 'red'

    if threshold is None:
        threshold = data[feature].median()

    if tags[0]:  # Se (True, False)
        data[feature + '_filter'] = np.where(data[feature] <= threshold, tags[0], tags[1])
    else:  # Se (False, True)
        data[feature + '_filter'] = np.where(data[feature] < threshold, tags[0], tags[1])

    thresholds = range(0, int(data[feature].max()) + 1, 1)

    # Calculate dataset sizes for <= and >= threshold conditions
    dataset_sizes_less_equal = []
    dataset_sizes_greater_equal = []

    closest_threshold = min(thresholds, key = lambda x: abs(x - threshold))

    for threshold_vals in thresholds:
        dataset_sizes_less_equal.append(len(data[data[feature] <= threshold_vals]))
        dataset_sizes_greater_equal.append(len(data[data[feature] >= threshold_vals]))

    # if add_column:
    # data[feature + '_filter'] = np.where(data[feaure].str.contains('Residencial'), 1, 0)

    if plot_analsis:
        # Set up side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize = (16, 6))

        # Plot 1: Dataset sizes for <= and >= thresholds
        axes[0].plot(thresholds, dataset_sizes_less_equal, color = color_A, linestyle = '-', label = f'<= {threshold}')
        axes[0].plot(thresholds, dataset_sizes_greater_equal, color = color_B, linestyle = '-',
                     label = f'>= {threshold}')
        axes[0].axvline(threshold, color = 'black', lw = 3, label = f'Threshold')

        # Retrieve the y-values at vline_threshold for the labels
        y_value_less_equal = dataset_sizes_less_equal[thresholds.index(threshold)]
        y_value_greater_equal = dataset_sizes_greater_equal[thresholds.index(threshold)]

        # Add labels at intersection points
        axes[0].text(threshold, y_value_less_equal, f'{y_value_less_equal}', color = color_A,
                     verticalalignment = 'bottom', horizontalalignment = 'right',
                     bbox = dict(facecolor = 'white', edgecolor = color_A, boxstyle = 'round,pad=0.3'))

        axes[0].text(threshold, y_value_greater_equal, f'{y_value_greater_equal}', color = color_B,
                     verticalalignment = 'top', horizontalalignment = 'right',
                     bbox = dict(facecolor = 'white', edgecolor = color_B, boxstyle = 'round,pad=0.3'))

        # Set x-axis limits for the KDE plot
        if xlim is not None:
            axes[0].set_xlim(xlim)

        # Set titles and labels based on plot_map
        title_text1 = plot_map.get('title_text1', 'Dataset Size for Thresholds').format(feature = feature)
        axes[0].set_title(f"{title_text1}", fontsize = plot_map.get('title_fontsize', 18))
        axes[0].set_xlabel(plot_map.get('x_label_text', 'Threshold for {}'.format(feature)),
                           fontsize = plot_map.get('label_fontsize', 16))
        axes[0].set_ylabel(plot_map.get('y_label_text0', 'Dataset Size (Number of Rows)'),
                           fontsize = plot_map.get('label_fontsize', 16))
        axes[0].grid(True)
        axes[0].legend(fontsize = plot_map.get('legend_fontsize', 14))
        axes[0].tick_params(labelsize = plot_map.get('tick_fontsize', 14))

        # KDE Plot with layered fills
        sns.kdeplot(data[feature], color = 'grey', ax = axes[1])

        # KDE plot for area to the left of vline_threshold (<= threshold)
        sns.kdeplot(data[feature], color = color_A, fill = True, ax = axes[1],
                    label = f'<= {threshold}', clip = (-np.inf, threshold), alpha = 0.3)

        # KDE plot for area to the right of vline_threshold (>= threshold)
        sns.kdeplot(data[feature], color = color_B, fill = True, ax = axes[1],
                    label = f'>= {threshold}', clip = (threshold, np.inf), alpha = 0.3)

        # Add vertical line at vline_threshold
        axes[1].axvline(threshold, color = 'black', lw = 3, label = f'Threshold')

        # Calculate the density at the vline_threshold
        kde = gaussian_kde(data[feature])
        density_at_vline = kde(threshold)

        # Calculate proportions
        total_count = len(data)
        proportion_less_equal = len(data[data[feature] <= threshold]) / total_count
        proportion_greater_equal = len(data[data[feature] >= threshold]) / total_count

        # Add labels for proportions at the KDE plot
        axes[1].text(threshold, density_at_vline, f'{proportion_less_equal:.2%}', color = color_A,
                     verticalalignment = 'bottom', horizontalalignment = 'right',
                     bbox = dict(facecolor = 'white', edgecolor = color_A, boxstyle = 'round,pad=0.3'))

        axes[1].text(threshold, density_at_vline * 0.8, f'{proportion_greater_equal:.2%}', color = color_B,
                     verticalalignment = 'top', horizontalalignment = 'right',
                     bbox = dict(facecolor = 'white', edgecolor = color_B, boxstyle = 'round,pad=0.3'))

        # Set x-axis limits for the KDE plot
        if xlim is not None:
            axes[1].set_xlim(xlim)

        title_text2 = plot_map.get('title_text2', 'Dataset Size for Thresholds').format(feature = feature)
        # Final adjustments for KDE plot
        axes[1].set_title(title_text2, fontsize = plot_map.get('title_fontsize', 18))
        axes[1].set_xlabel(plot_map.get('x_label_text', feature), fontsize = plot_map.get('label_fontsize', 16))
        axes[1].set_ylabel(plot_map.get('y_label_text', 'Distribuição'), fontsize = plot_map.get('label_fontsize', 16))
        axes[1].legend(fontsize = plot_map.get('legend_fontsize', 14))
        axes[1].tick_params(labelsize = plot_map.get('tick_fontsize', 14))

        plt.tight_layout()
        plt.show()

    if filter_data:
        data = data[data[feature + '_filter'] == True].copy()
        data.drop(columns = feature + '_filter', inplace = True)
    return data



def create_labels(bins):
    labels = []
    for i in range(len(bins) - 1):
        if bins[i + 1] == np.inf:
            labels.append(f'{bins[i]}+')
        else:
            labels.append(f'{bins[i]}-{bins[i + 1]}')
    return labels


def plot_time_series_with_resampling(data, tgt_val, granularity, agg_func = 'mean', date_range = None, bins = None):
    df = data.sort_values(by = 'data_lancamento').copy()

    df['data_lancamento'] = pd.to_datetime(df['data_lancamento'])

    # Set the 'data_lancamento' column as the index
    df.set_index('data_lancamento', inplace = True)

    if date_range:
        start_date, end_date = date_range
        df = df[start_date:end_date]

    # Dictionary to map string function names to actual pandas functions
    agg_funcs = {
        'mean': 'mean',
        'count': 'count',
        'sum': 'sum',
        'min': 'min',
        'max': 'max'
        }

    # Check if the provided agg_func is valid
    if agg_func not in agg_funcs:
        raise ValueError(f"Invalid agg_func. Choose from {list(agg_funcs.keys())}")

    # Resample the data using the specified aggregation function
    resampled_data = df[tgt_val].resample(granularity).agg(agg_funcs[agg_func]).reset_index()

    # Create a plot
    fig, ax = plt.subplots(1, 1, figsize = (12, 6))

    # Create bins for the tgt_val column
    if bins is None:
        bins = [0, 50, 100, 200, 300, np.inf]

    labels = create_labels(bins)
    df['m2_tipo'] = pd.cut(df[tgt_val], bins = bins, labels = labels)
    aux_df = df[['m2_tipo'] + [tgt_val]].sort_values(by = 'm2_tipo')

    for bin in aux_df['m2_tipo'].unique():
        aux_bins = aux_df[aux_df['m2_tipo'] == bin]
        aux_monthly_mean = aux_bins[tgt_val].resample(granularity).agg(agg_funcs[agg_func]).reset_index()
        sns.lineplot(data = aux_monthly_mean, x = 'data_lancamento', y = tgt_val, label = bin, ax = ax)

    # Plot the overall resampled data
    sns.lineplot(data = resampled_data, x = 'data_lancamento', y = tgt_val,
                 label = f'{agg_func.capitalize()} of {tgt_val}', color = 'black',
                 lw = 2.5, ls = 'dashed', ax = ax)

    # Customize the plot
    plt.title('Time Series Data and Resampled Data')
    plt.xlabel('Date')
    plt.ylabel('Quantidade lançamentos')
    plt.legend()
    plt.show()


def boxplots_idade(data_full, data_nan, bins = None):
    if bins is None:
        bins = [0, 50, 100, 200, 300, np.inf]

    labels = create_labels(bins)

    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize = (14, 6))

    # First boxplot
    sns.boxplot(data = data_full, x = 'Idade_predio',
                hue = pd.cut(data_full['M2_total_unidade_tipo'],
                             bins = bins,
                             labels = labels),
                gap = 0.2, ax = axes[0])  # Specify the second subplot axis

    # Set title for the first plot
    axes[0].set_title(f'Idade de todos os prédios da base ({data_full.shape[0]})')
    axes[0].get_legend().remove()
    # Second boxplot
    sns.boxplot(data = data_nan, x = 'Idade_predio',
                hue = pd.cut(data_nan['M2_total_unidade_tipo'],
                             bins = bins,
                             labels = labels),
                gap = 0.2, ax = axes[1])  # Specify the first subplot axis

    # Set only observed values on x-axis
    observed_values = data_nan['Idade_predio'].dropna().unique()
    axes[1].set_xticks(observed_values)

    # Move the legend outside the figure for the second plot
    axes[1].legend(title = 'M2_total_unidade_tipo', bbox_to_anchor = (1.05, 1), loc = 'upper left')

    # Set title for the second plot
    axes[1].set_title(f'Idade dos prédios com valores faltantes de metragem ({data_nan.shape[0]}/{data_full.shape[0]})')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def plot_building_age_density(df, age_column = 'Idade_predio', area_column = 'M2_total_unidade_tipo'):
    # Define bins and create labels
    bins = [0, 50, 100, 200, 300, np.inf]
    labels = create_labels(bins)

    # Print proportion of new buildings
    print(f"Proporção prédios novos (>2000) / Total:\t{len(df[df[age_column] < 25])} / {len(df)}")

    # Filter data for buildings with Idade_predio < 25
    aux_clean = df[df[age_column] < 25]

    # Plot setup
    plt.figure(figsize = (14, 6))

    # Add shaded areas with lighter colors
    plt.axvspan(0, 10, color = 'lightblue', alpha = 0.3, )
    plt.axvspan(10, 17, color = 'lightgreen', alpha = 0.3, )
    plt.axvspan(17, aux_clean[age_column].max(), color = 'lightcoral', alpha = 0.3, )
    # Axis controls

    # KDE plot with hue, capturing line elements for legend
    sns.kdeplot(
        x = aux_clean[age_column],
        hue = pd.cut(aux_clean[area_column], bins = bins, labels = labels),
        lw = 1.5, palette = 'Set1'
        )

    plt.xlim(0, aux_clean[age_column].max())  # Set x-axis range
    plt.ylim(0, None)  # Let y-axis auto adjust, or set a specific limit if needed

    # Labels and legend
    plt.xlabel("Idade do Prédio")
    plt.ylabel("Density")

    # Add grid and display the plot
    plt.grid(True)
    plt.show()