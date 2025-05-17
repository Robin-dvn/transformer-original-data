"""
This module provides a SimpleValidator class for comparing sequence datasets from different sources.
It includes methods for statistical analysis and visualization of sequence properties.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

# Configure default renderer for Plotly
pio.renderers.default = 'browser'


class SimpleValidator:
    """
    A validator for comparing sequence datasets from different sources.

    This class provides methods to validate and compare sequence datasets,
    analyzing their statistical properties and visualizing the differences.
    """

    def __init__(self, show_plots=True):
        """
        Initialize the SimpleValidator.

        Args:
            show_plots (bool): Whether to display plots during analysis.
        """
        self.stats = {}
        self.show = show_plots

    def filter_and_include_all_if_present(self, csv_paths, dataset_names):
        """
        Filter data to keep only pairs (Observation, Year) present in ALL datasets,
        and include ALL sequences for these pairs without equalization.

        Args:
            csv_paths (list): List of paths to CSV files.
            dataset_names (list): List of names corresponding to datasets.

        Returns:
            pd.DataFrame: Filtered DataFrame containing all sequences from
                         common pairs, with a 'Source' column.
        """
        dataframes = [pd.read_csv(path, dtype={"Sequence": str}) for path in csv_paths]

        for df in dataframes:
            # Ensure column types
            df['Observation'] = df['Observation'].astype(str)
            df['Year'] = df['Year'].astype(str)

        # Find unique (Observation, Year) pairs in each dataframe
        pairs_per_df = [
            set(df[['Observation', 'Year']].drop_duplicates().apply(tuple, axis=1))
            for df in dataframes
        ]

        # Find intersection: pairs present in ALL dataframes
        common_pairs = set.intersection(*pairs_per_df)

        # Initialize the final DataFrame
        filtered_df = pd.DataFrame()

        # Iterate over common pairs
        for obs, year in common_pairs:
            # Get ALL rows for this pair from each dataframe
            subsets_to_concat = []
            all_present = True  # Flag to verify presence in all DFs after potential filtering
            for df, name in zip(dataframes, dataset_names):
                subset = df[(df['Observation'] == obs) & (df['Year'] == year)].copy()
                # Additional check: ensure there is data for this pair in this specific dataframe
                if subset.empty:
                    all_present = False
                    break  # If a DF has nothing for this pair, ignore it completely
                subset['Source'] = name
                subsets_to_concat.append(subset)

            # If the pair is present and non-empty in all dataframes
            if all_present:
                # Concatenate all subsets for this pair
                filtered_df = pd.concat([filtered_df] + subsets_to_concat, ignore_index=True)

        return filtered_df

    def sequence_length_validation(self, dataframe):
        """
        Validate sequence lengths across different sources.

        Compares the mean and standard deviation of sequence lengths for each
        (Observation, Year) pair across different sources.

        Args:
            dataframe (pd.DataFrame): DataFrame containing sequences to analyze
                                      with a 'Source' column to distinguish origins.
        """
        dataframe = dataframe[dataframe["Sequence"] != "0"]

        unique_pairs = dataframe[['Observation', 'Year']].drop_duplicates()

        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']
            subset = dataframe[(dataframe['Observation'] == observation) &
                               (dataframe['Year'] == year)].copy()

            subset['Sequence Length'] = subset['Sequence'].apply(len)
            stats = subset.groupby('Source')['Sequence Length'].agg(['mean', 'std']).reset_index()

            # Handle cases where std might be NaN (if a group has a single sequence)
            stats['std'] = stats['std'].fillna(0)

            # Calculate y limits accounting for potentially null std
            max_std = stats['std'].max()
            y_max = stats['mean'].max() + (10 * max_std if max_std > 0 else 10)
            y_min = stats['mean'].min() - (10 * max_std if max_std > 0 else 10)
            # Ensure y_min isn't negative if lengths are always positive
            y_min = max(0, y_min)

            # Create the plot
            fig = px.scatter(stats, x='Source', y='mean', error_y='std',
                             title=f'Sequence Length Comparison for {observation} in {year}',
                             labels={'mean': 'Average Sequence Length', 'std': 'Standard Deviation'},
                             color='Source',
                             size_max=10)
            fig.update_traces(marker=dict(size=12, opacity=0.8), error_y=dict(width=5))
            fig.update_yaxes(range=[y_min, y_max])
            fig.update_layout(
                legend=dict(x=1.02, y=1, font=dict(size=12)),
                margin=dict(l=50, r=200, t=50, b=50),
                title_text=f'Sequence length for {observation} in {year}',
                height=800,
                width=1200
            )

            if self.show:
                fig.show()

    def sequence_length_distribution_validation(self, dataframe):
        """
        Analyze the distribution of sequence lengths using histograms and Jensen-Shannon distance.

        Args:
            dataframe (pd.DataFrame): DataFrame containing sequences to analyze
                                     with a 'Source' column to distinguish origins.
        """
        # Filter empty sequences
        dataframe = dataframe[dataframe["Sequence"] != "0"]

        unique_pairs = dataframe[['Observation', 'Year']].drop_duplicates()

        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']
            subset = dataframe[(dataframe['Observation'] == observation) &
                              (dataframe['Year'] == year)].copy()

            # Calculate the length of each sequence
            subset['Sequence Length'] = subset['Sequence'].apply(len)

            # Separate data by source
            sources = subset['Source'].unique()

            # Create figure for histogram
            fig = go.Figure()

            # Variables to store KDE distributions
            kde_norm_values = {}
            x_grids = {}  # Store x grids for each source

            # Collect lengths by source
            all_lengths = subset['Sequence Length'].values
            if len(all_lengths) == 0:
                continue  # Skip if no data for this pair
            global_min_len = min(all_lengths)
            global_max_len = max(all_lengths)

            for source in sources:
                # Extract lengths for this source
                lengths = subset[subset['Source'] == source]['Sequence Length'].values
                if len(lengths) == 0:
                    continue  # Skip if no data for this source

                # Add histogram for this source
                fig.add_trace(go.Histogram(
                    x=lengths,
                    name=source,
                    opacity=0.6,
                    histnorm='probability density'
                ))

                # Calculate KDE if possible (at least 2 points and variance > 0)
                if len(lengths) > 1 and np.var(lengths) > 0:
                    try:
                        # Define a common x grid based on global range
                        x_grid = np.linspace(global_min_len, global_max_len, 500)  # Common grid
                        x_grids[source] = x_grid

                        # Calculate KDE
                        kde = gaussian_kde(lengths, bw_method='scott')
                        kde_values = kde(x_grid)

                        # Normalize for Jensen-Shannon
                        dx = (global_max_len - global_min_len) / (len(x_grid) - 1) if len(x_grid) > 1 else 1
                        kde_norm = kde_values / np.sum(kde_values * dx)
                        kde_norm_values[source] = kde_norm

                        # Add KDE curve
                        fig.add_trace(go.Scatter(
                            x=x_grid,
                            y=kde_norm,
                            mode='lines',
                            name=f'KDE {source}'
                        ))
                    except Exception as e:
                        print(f"Error calculating KDE for {source} ({observation}, {year}): {e}")
                        # Ensure source is not in kde_norm_values if calculation fails
                        if source in kde_norm_values:
                            del kde_norm_values[source]
                        if source in x_grids:
                            del x_grids[source]

            # Calculate Jensen-Shannon distances between sources with valid KDE
            valid_sources = list(kde_norm_values.keys())
            js_distances_text = []  # For display

            for i in range(len(valid_sources)):
                for j in range(i+1, len(valid_sources)):
                    source1 = valid_sources[i]
                    source2 = valid_sources[j]

                    # Ensure grids are compatible (they should be now)
                    if source1 in x_grids and source2 in x_grids and len(x_grids[source1]) == len(x_grids[source2]):
                        # Calculate JS distance
                        # Add a small epsilon to avoid log(0) if a density is zero
                        epsilon = 1e-10
                        p = kde_norm_values[source1] + epsilon
                        q = kde_norm_values[source2] + epsilon
                        js_distance = jensenshannon(p, q)

                        # Store in statistics
                        key = (observation, year, source1, source2)
                        if key not in self.stats:
                            self.stats[key] = {}
                        self.stats[key]["sequence_length_js_distance"] = js_distance
                        js_distances_text.append(f"{source1} vs {source2} JS: {js_distance:.4f}")
                    else:
                        print(f"Cannot calculate JS for {source1} vs {source2} ({observation}, {year}) due to uncalculated KDEs or incompatible grids.")

            # Final layout
            fig.update_layout(
                title=f"Sequence Length Distribution for {observation} in {year}",
                xaxis_title="Sequence Length",
                yaxis_title="Probability Density",
                barmode='overlay',  # Overlay histograms
                legend_title_text='Source',
                height=600,
                width=1000
            )
            fig.update_xaxes(range=[global_min_len, global_max_len])  # Set x-axis globally

            # Display the figure
            if self.show:
                fig.show()

    def sequence_digit_stats(self, dataframe, observation_filter=None, year_filter=None):
        """
        Analyze digit statistics in sequences and compare across datasets.

        Args:
            dataframe (pd.DataFrame): DataFrame containing sequences to analyze
                                      with a 'Source' column to distinguish origins.
            observation_filter (str, optional): Filter for specific observation type.
            year_filter (list, optional): Filter for specific years.
        """
        # Filter empty sequences
        dataframe = dataframe[dataframe["Sequence"] != "0"]

        # Apply filters if provided
        if observation_filter is not None:
            dataframe = dataframe[dataframe['Observation'] == observation_filter]
        if year_filter is not None:
            dataframe = dataframe[dataframe['Year'].isin(year_filter)]

        unique_pairs = dataframe[['Observation', 'Year']].drop_duplicates()

        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']
            subset = dataframe[(dataframe['Observation'] == observation) &
                              (dataframe['Year'] == year)].copy()

            sources = subset['Source'].unique()
            if len(sources) < 2:
                continue  # No comparison possible if less than 2 sources

            # For each digit (0-4), create a dedicated graph
            for digit in range(5):
                # Calculate statistics for each source
                stats_data = []

                for source in sources:
                    source_data = subset[subset['Source'] == source]
                    if source_data.empty:
                        continue  # Skip if no data for this source/pair

                    # Count occurrences of digit in each sequence
                    # Ensure seq is a string before counting
                    counts = [str(seq).count(str(digit)) for seq in source_data['Sequence'] if pd.notna(seq)]
                    if not counts:  # If no valid sequences found
                        mean_count = 0
                        std_count = 0
                    else:
                        mean_count = np.mean(counts)
                        std_count = np.std(counts)  # std of 0 if len(counts) == 1

                    stats_data.append({
                        'Source': source,
                        'Mean': mean_count,
                        'Std': std_count
                    })

                # If we couldn't calculate stats for at least 2 sources, skip to next digit
                if len(stats_data) < 2:
                    continue

                # Create DataFrame for the plot
                digit_stats_df = pd.DataFrame(stats_data)

                # Calculate errors between sources
                errors_text = []
                for i in range(len(sources)):
                    for j in range(i+1, len(sources)):
                        source1 = sources[i]
                        source2 = sources[j]

                        # Find corresponding stats in digit_stats_df
                        stats1_row = digit_stats_df[digit_stats_df['Source'] == source1]
                        stats2_row = digit_stats_df[digit_stats_df['Source'] == source2]

                        # Ensure both sources have data
                        if stats1_row.empty or stats2_row.empty:
                            continue

                        stats1 = stats1_row.iloc[0]
                        stats2 = stats2_row.iloc[0]

                        mean_error = abs(stats1['Mean'] - stats2['Mean'])
                        std_error = abs(stats1['Std'] - stats2['Std'])

                        # Store errors in self.stats dictionary
                        key = (observation, year, source1, source2)
                        if key not in self.stats:
                            self.stats[key] = {}

                        if "digit_mean_errors" not in self.stats[key]:
                            self.stats[key]["digit_mean_errors"] = {}
                        if "digit_std_errors" not in self.stats[key]:
                            self.stats[key]["digit_std_errors"] = {}

                        self.stats[key]["digit_mean_errors"][digit] = mean_error
                        self.stats[key]["digit_std_errors"][digit] = std_error
                        errors_text.append(f"{source1} vs {source2}: Mean Err: {mean_error:.2f}, Std Err: {std_error:.2f}")

                # Create bar chart for this digit
                fig = px.bar(
                    digit_stats_df,
                    x='Source',
                    y='Mean',
                    error_y='Std',
                    title=f'Digit {digit} Occurrence Stats - {observation} in {year}',
                    labels={'Mean': f'Avg Occurrences of Digit {digit}', 'Std': 'Standard Deviation'},
                    color='Source'
                )

                # Add a single annotation for errors
                if errors_text:
                    fig.add_annotation(
                        x=0.98, y=0.98,
                        xref="paper", yref="paper",
                        text="<br>".join(errors_text),
                        showarrow=False,
                        align="right",
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4,
                        opacity=0.8
                    )

                # Plot layout
                fig.update_layout(
                    xaxis_title="Source",
                    yaxis_title=f"Average occurrences of digit {digit}",
                    height=600,
                    width=800,
                    margin=dict(l=50, r=250, t=80, b=50)  # Increase right margin for annotation
                )

                if self.show:
                    fig.show()

    def count_series(self, sequence, digit):
        """
        Count the number of series of a given digit and their average size.

        A series is defined as consecutive occurrences of the same digit.

        Args:
            sequence (str): The sequence to analyze
            digit (int): The digit to find series of

        Returns:
            tuple: (num_series, mean_length, std_length)
        """
        series_lengths = []
        current_series_length = 0
        digit_char = str(digit)  # Convert digit to character once

        for char in sequence:
            if char == digit_char:
                current_series_length += 1
            else:
                if current_series_length > 0:
                    series_lengths.append(current_series_length)
                current_series_length = 0  # Reset if character changes

        # Add the last series if sequence ends with the target digit
        if current_series_length > 0:
            series_lengths.append(current_series_length)

        # Calculate statistics on series lengths found
        num_series = len(series_lengths)
        if num_series == 0:
            mean_len = 0
            std_len = 0
        else:
            mean_len = np.mean(series_lengths)
            # Calculate std only if more than one series exists
            std_len = np.std(series_lengths) if num_series > 1 else 0

        return num_series, mean_len, std_len

    def sequence_series_analysis(self, dataframe, observation_filter=None, year_filter=None):
        """
        Analyze digit series in sequences and compare statistics across datasets.

        A series is defined as consecutive occurrences of the same digit in a sequence.

        Args:
            dataframe (pd.DataFrame): DataFrame containing sequences to analyze
                                      with a 'Source' column to distinguish origins.
            observation_filter (str, optional): Filter for specific observation type.
            year_filter (list, optional): Filter for specific years.
        """
        # Filter empty or invalid sequences
        dataframe = dataframe[dataframe["Sequence"].notna() & (dataframe["Sequence"] != "0")]
        # Ensure Sequence is string type for iteration
        dataframe['Sequence'] = dataframe['Sequence'].astype(str)

        # Apply filters if provided
        if observation_filter is not None:
            dataframe = dataframe[dataframe['Observation'] == observation_filter]
        if year_filter is not None:
            dataframe = dataframe[dataframe['Year'].isin(year_filter)]

        unique_pairs = dataframe[['Observation', 'Year']].drop_duplicates()

        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']
            subset = dataframe[(dataframe['Observation'] == observation) &
                              (dataframe['Year'] == year)].copy()

            sources = subset['Source'].unique()
            if len(sources) < 2:
                continue  # Need at least two sources to compare

            # Analysis for each digit (0-4)
            for digit in range(5):
                # Dictionary to store aggregated statistics by source
                source_aggregated_stats = {}

                # Calculate statistics for each source
                valid_sources_for_digit = []
                for source in sources:
                    source_subset = subset[subset['Source'] == source]
                    if source_subset.empty:
                        continue

                    # Lists to store results per sequence for this source
                    all_num_series = []
                    all_mean_lengths = []
                    all_std_lengths = []

                    for seq in source_subset['Sequence']:
                        # Ensure seq is a non-empty string
                        if isinstance(seq, str) and seq:
                            num, mean_len, std_len = self.count_series(seq, digit)
                            all_num_series.append(num)
                            # Add mean and std only if at least one series was found (mean/std > 0)
                            if num > 0:
                                all_mean_lengths.append(mean_len)
                                all_std_lengths.append(std_len)

                    # Calculate aggregated statistics for the source
                    # Average number of series per sequence
                    count_mean = np.mean(all_num_series) if all_num_series else 0
                    count_std = np.std(all_num_series) if len(all_num_series) > 1 else 0
                    # Average of mean series sizes (over sequences that had series)
                    mean_mean = np.mean(all_mean_lengths) if all_mean_lengths else 0
                    mean_std = np.std(all_mean_lengths) if len(all_mean_lengths) > 1 else 0
                    # Average of standard deviations of series sizes (over sequences that had series)
                    std_mean = np.mean(all_std_lengths) if all_std_lengths else 0
                    std_std = np.std(all_std_lengths) if len(all_std_lengths) > 1 else 0

                    source_aggregated_stats[source] = {
                        'count_mean': count_mean, 'count_std': count_std,
                        'mean_mean': mean_mean, 'mean_std': mean_std,
                        'std_mean': std_mean, 'std_std': std_std
                    }
                    valid_sources_for_digit.append(source)

                # If fewer than two sources have valid data for this digit, skip to next
                if len(valid_sources_for_digit) < 2:
                    continue

                # Calculate errors between all pairs of valid sources
                errors_text = []
                current_sources = valid_sources_for_digit  # Use sources that actually have data
                for i in range(len(current_sources)):
                    for j in range(i+1, len(current_sources)):
                        source1 = current_sources[i]
                        source2 = current_sources[j]

                        stats1 = source_aggregated_stats[source1]
                        stats2 = source_aggregated_stats[source2]

                        count_error = abs(stats1['count_mean'] - stats2['count_mean'])
                        mean_error = abs(stats1['mean_mean'] - stats2['mean_mean'])
                        std_error = abs(stats1['std_mean'] - stats2['std_mean'])

                        # Update global statistics
                        key = (observation, year, source1, source2)
                        if key not in self.stats:
                            self.stats[key] = {}
                        if f"digit_{digit}_series_errors" not in self.stats[key]:
                            self.stats[key][f"digit_{digit}_series_errors"] = {}

                        self.stats[key][f"digit_{digit}_series_errors"]['count_error'] = count_error
                        self.stats[key][f"digit_{digit}_series_errors"]['mean_error'] = mean_error
                        self.stats[key][f"digit_{digit}_series_errors"]['std_error'] = std_error
                        errors_text.append(f"{source1} vs {source2}: CntErr:{count_error:.2f}, MeanErr:{mean_error:.2f}, StdErr:{std_error:.2f}")

                # Create a graph to visualize results
                fig = go.Figure()

                # Prepare data for grouped bar chart
                plot_data = {'Source': [], 'Metric': [], 'Value': [], 'StdDev': []}
                metrics_map = {
                    'Average Series Count': ('count_mean', 'count_std'),
                    'Average Series Size': ('mean_mean', 'mean_std'),
                    'Avg Std of Series Sizes': ('std_mean', 'std_std')
                }

                for source in current_sources:
                    stats = source_aggregated_stats[source]
                    for metric_name, (mean_key, std_key) in metrics_map.items():
                         plot_data['Source'].append(source)
                         plot_data['Metric'].append(metric_name)
                         plot_data['Value'].append(stats[mean_key])
                         plot_data['StdDev'].append(stats[std_key])

                plot_df = pd.DataFrame(plot_data)

                # Create grouped chart
                fig = px.bar(plot_df, x='Metric', y='Value', color='Source',
                             barmode='group', error_y='StdDev',
                             title=f"Digit {digit} Series Statistics - {observation} {year}",
                             labels={'Value': 'Average Value', 'Metric': 'Statistic', 'StdDev': 'Standard Deviation'})

                # Add annotations for errors between pairs
                if errors_text:
                    fig.add_annotation(
                        x=1.0, y=1.0,  # Position at top right
                        xref="paper", yref="paper",
                        text="Absolute Mean Errors:<br>" + "<br>".join(errors_text),
                        showarrow=False,
                        align="right",
                        font=dict(size=10),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4
                    )

                fig.update_layout(
                    height=700,  # Adjust height if needed
                    width=1200,
                    margin=dict(t=100, b=100, l=50, r=300),  # Increase right margin for annotations
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                if self.show:
                    fig.show()

    def print_stats_summary(self, max_items=5):
        """
        Print a summary of the collected statistics.

        Args:
            max_items (int): Maximum number of items to show in the summary.
        """
        print("\nCollected Statistics (Overview):")
        count = 0
        for key, value in self.stats.items():
            if len(key) == 4:  # Format (observation, year, source1, source2)
                print(f"\n{key[0]} - {key[1]}, {key[2]} vs {key[3]}:")
                for stat_name, stat_value in value.items():
                    # Handle nested dictionaries (for errors by digit)
                    if isinstance(stat_value, dict):
                        print(f"  {stat_name}:")
                        details = []
                        for sub_key, sub_value in stat_value.items():
                             # Try to format as float, otherwise display as is
                             try:
                                 details.append(f"{sub_key}: {sub_value:.3f}")
                             except (TypeError, ValueError):
                                 details.append(f"{sub_key}: {sub_value}")
                        print(f"    {', '.join(details)}")
                    else:
                        # Try to format as float, otherwise display as is
                        try:
                            print(f"  {stat_name}: {stat_value:.3f}")
                        except (TypeError, ValueError):
                            print(f"  {stat_name}: {stat_value}")
                count += 1
                if count >= max_items:  # Limit the overview to avoid overloading output
                    print("\n[... more statistics ...]")
                    break
        if not self.stats:
            print("No comparison statistics were calculated (maybe no common pairs or not enough data).")


def main():
    """
    Main function to demonstrate the SimpleValidator usage.
    """
    # Create validator
    validator = SimpleValidator(show_plots=True)

    # Example paths (relative paths are preferred over absolute)
    dataset_paths = [
        "./experiments/DO_NBL-15_DM-32_DFF-1024_TS-20250422-091237_optuna/generated_dataset.csv",
        "./experiments/DO_NBL-15_DM-32_DFF-1024_TS-20250423-143325/generated_dataset.csv",
        "./markov_python_generated_dataset10000.csv",
        "./data/all_sequences.csv"
    ]

    dataset_names = [
        "transformer_model_1",
        "transformer_model_2",
        "markov",
        "original_data"
    ]

    # Use the new filtering/inclusion method
    df = validator.filter_and_include_all_if_present(dataset_paths, dataset_names)
    print(f"Filtered DataFrame contains {len(df)} rows.")
    print("Distribution by source:")
    print(df['Source'].value_counts())
    print("\nPairs (Observation, Year) present:")
    print(df[['Observation', 'Year']].drop_duplicates())

    # Call analysis methods
    validator.sequence_length_validation(df.copy())
    validator.sequence_length_distribution_validation(df.copy())
    validator.sequence_digit_stats(df.copy(), observation_filter='MEDIUM', year_filter=['Y4', 'Y5'])
    validator.sequence_series_analysis(df.copy())

    # Show collected statistics summary
    validator.print_stats_summary(max_items=5)


if __name__ == "__main__":
    main()
