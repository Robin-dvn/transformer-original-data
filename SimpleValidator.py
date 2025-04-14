import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
import webbrowser
webbrowser.register('firefox', None, webbrowser.BackgroundBrowser('/usr/bin/firefox'))


class SimpleValidator:
    def __init__(self) -> None:
        self.stats = {}
        self.show = True

    def filter_and_equalize_multiple_datasets(self, csv_paths, dataset_names):
        dataframes = [pd.read_csv(path, dtype={"Sequence": str}) for path in csv_paths]

        for df in dataframes:
            df['Observation'] = df['Observation'].astype(str)
            df['Year'] = df['Year'].astype(str)

        common_pairs = set.intersection(*[
            set(df[['Observation', 'Year']].drop_duplicates().apply(tuple, axis=1))
            for df in dataframes
        ])

        filtered_df = pd.DataFrame()
        for obs, year in common_pairs:
            subsets = [df[(df['Observation'] == obs) & (df['Year'] == year)] for df in dataframes]
            min_len = min(len(s) for s in subsets)
            for df, name in zip(subsets, dataset_names):
                subset = df.head(min_len).copy()
                subset['Source'] = name
                filtered_df = pd.concat([filtered_df, subset])
        return filtered_df

    def sequence_length_validation(self, dataframe):
        dataframe = dataframe[dataframe["Sequence"] != "0"]
        dataframe = dataframe[(dataframe['Observation'] == 'MEDIUM') & (dataframe['Year'].isin(['Y4', 'Y5']))]

        unique_pairs = dataframe[['Observation', 'Year']].drop_duplicates()

        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']
            subset = dataframe[(dataframe['Observation'] == observation) &
                               (dataframe['Year'] == year)].copy()

            subset['Sequence Length'] = subset['Sequence'].apply(len)
            stats = subset.groupby('Source')['Sequence Length'].agg(['mean', 'std']).reset_index()

            y_max = stats['mean'].max() + 10 * stats['std'].max()
            y_min = stats['mean'].min() - 10 * stats['std'].max()

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


if __name__ == "__main__":
    validator = SimpleValidator()
    validator.show = True
    dataset_paths = [
        "experiments/DO_NBL-15_DM-32_DFF-1024_TS-20250414-101844/generated_dataset.csv",
        "markov_python_generated_dataset10000.csv",
        'data/all_sequences.csv'
    ]
    dataset_names = [
        "exp trans",
        "markov",
        "original_data"
    ]
    df = validator.filter_and_equalize_multiple_datasets(dataset_paths, dataset_names)
    validator.sequence_length_validation(df)
