
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
import webbrowser
webbrowser.register('firefox', None, webbrowser.BackgroundBrowser('/usr/bin/firefox'))


class SimpleValidator:
    def __init__(self) -> None:
        self.stats={}
        self.show=True


    def filter_and_equalize_dataset(self, csv_path_1, csv_path_2,dataset_name_1,dataset_name_2):
        # Charger les deux datasets dans des DataFrames
        df1 = pd.read_csv(csv_path_1,dtype={"Sequence":str})  # Remplace par le chemin vers ton premier fichier
        df2 = pd.read_csv(csv_path_2,dtype={"Sequence":str})  # Remplace par le chemin vers ton deuxième fichier

        # Trouver les paires uniques (Observation, Year) dans chaque dataset
        df1_unique = df1[['Observation', 'Year']].drop_duplicates()
        df2_unique = df2[['Observation', 'Year']].drop_duplicates()

        # Trouver les paires communes
        common_pairs = pd.merge(df1_unique, df2_unique, on=['Observation', 'Year'])

        # Créer un DataFrame vide pour les lignes filtrées
        filtered_df = pd.DataFrame()

        # Pour chaque paire commune, garder les n premières lignes où n est le minimum
        for _, pair in common_pairs.iterrows():
            observation = pair['Observation']
            year = pair['Year']

            # Nombre de lignes à conserver (minimum entre df1 et df2 pour cette paire)
            count_df1 = len(df1[(df1['Observation'] == observation) & (df1['Year'] == year)])
            count_df2 = len(df2[(df2['Observation'] == observation) & (df2['Year'] == year)])

            # Nombre de lignes à conserver pour cette paire
            num_to_keep = min(count_df1, count_df2)
            print(f"Observation: {observation}, Year: {year}, Num to keep: {num_to_keep}")
            # Ajouter les lignes correspondantes aux deux DataFrames avec la colonne 'source'
            df1_subset = df1[(df1['Observation'] == observation) & (df1['Year'] == year)].head(num_to_keep)
            df2_subset = df2[(df2['Observation'] == observation) & (df2['Year'] == year)].head(num_to_keep)

            # Ajouter la colonne 'source' pour chaque DataFrame
            df1_subset['Source'] = dataset_name_1
            df2_subset['Source'] = dataset_name_2

            # Concatenation des lignes filtrées avec la colonne source
            filtered_df = pd.concat([filtered_df, df1_subset, df2_subset])
        return filtered_df

    def sequence_length_validation(self, dataframe):
        """
        Compare la longueur des séquences générées avec celles du dataset original.

        Args:
            dataframe (pd.DataFrame): DataFrame contenant les données générées.
        """

        dataframe = dataframe[dataframe["Sequence"] !=0]
        dataframe = dataframe[(dataframe['Observation'] == 'MEDIUM') & (dataframe['Year'].isin(['Y4','Y5']))]

        unique_pairs = dataframe[['Observation', 'Year']].drop_duplicates()

        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']
            subset = dataframe[(dataframe['Observation'] == observation) &
                                    (dataframe['Year'] == year)].copy()


            subset['Sequence Length'] = subset['Sequence'].apply(len)
            stats = subset.groupby('Source')['Sequence Length'].agg(['mean', 'std']).reset_index()

            mean_error = abs(stats.loc[stats['Source'] == 'Dataset', 'mean'].values[0] -
                            stats.loc[stats['Source'] == 'Generated Dataset', 'mean'].values[0])
            std_error = abs(stats.loc[stats['Source'] == 'Dataset', 'std'].values[0] -
                            stats.loc[stats['Source'] == 'Generated Dataset', 'std'].values[0])
            # self.stats[(observation, year)]["mean_error"] = mean_error


            y_max = stats['mean'].max() + 10 * stats['std'].max()
            y_min = stats['mean'].min() - 10 * stats['std'].max()

            # On retire les annotations sur les points en n'utilisant pas le paramètre 'text'
            fig = px.scatter(stats, x='Source', y='mean', error_y='std',
                            title=f'Sequence Length Comparison for {observation} in {year}',
                            labels={'mean': 'Average Sequence Length', 'std': 'Standard Deviation'},
                            color='Source',
                            color_discrete_map={'Dataset': 'green', 'Generated Dataset': 'blue'},
                            size_max=10)
            fig.update_traces(marker=dict(size=12, opacity=0.8), error_y=dict(width=5))
            fig.update_yaxes(range=[y_min, y_max])
            fig.update_xaxes(range=[-1, 2])

            # Positionner la légende à droite et agrandir la marge droite pour faire de la place
            fig.update_layout(
                legend=dict(x=1.02, y=1, font=dict(size=12)),
                margin=dict(l=50, r=200, t=50, b=50)
            )
            # Combiner les stats dans une annotation unique formatée avec des retours à la ligne
            annotation_text = (
                f"Dataset<br>Mean: {stats.loc[stats['Source']=='Dataset', 'mean'].values[0]:.2f}<br>"
                f"Std: {stats.loc[stats['Source']=='Dataset', 'std'].values[0]:.2f}<br><br>"
                f"Generated Dataset<br>Mean: {stats.loc[stats['Source']=='Generated Dataset', 'mean'].values[0]:.2f}<br>"
                f"Std: {stats.loc[stats['Source']=='Generated Dataset', 'std'].values[0]:.2f}"
            )

            # Placer l'annotation en bas à droite du graph
            fig.update_layout(
                annotations=[
                    dict(
                        x=0.98, y=0.02, xref='paper', yref='paper',
                        text=annotation_text,
                        showarrow=False, font=dict(size=12),
                        xanchor='right', yanchor='bottom'
                    )
                ]
            )
            print("Graph generated successfully.")
            if self.show: fig.show()
            fig.update_layout(
                title_text=f'Sequence length for {observation} in {year}',
                height=800,
                width=1200,
                margin=dict(t=100, b=100, l=50, r=50)
            )



if __name__ == "__main__":
    validator = SimpleValidator()
    validator.show = True
    df = validator.filter_and_equalize_dataset("markov_python_generated_dataset10000.csv","data/all_sequences.csv","Generated Dataset","Dataset")
    validator.sequence_length_validation(df)
