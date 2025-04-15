import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
                
    def sequence_series_analysis(self, dataframe):
        """
        Analyse les séries de chiffres dans les séquences et compare les statistiques entre les datasets.
        Une série est définie comme des occurrences consécutives du même chiffre dans une séquence.
        
        Args:
            dataframe (pd.DataFrame): DataFrame contenant les séquences à analyser
                                      avec une colonne 'Source' pour distinguer l'origine.
        """
        # Filtrer les séquences vides
        dataframe = dataframe[dataframe["Sequence"] != "0"]
        
        unique_pairs = dataframe[['Observation', 'Year']].drop_duplicates()

        def count_series(sequence, digit):
            """Compte le nombre de séries d'un chiffre donné et leur taille moyenne."""
            series = []
            current_series = 0

            for char in sequence:
                if char == str(digit):
                    current_series += 1
                elif current_series > 0:
                    series.append(current_series)
                    current_series = 0

            # Ajouter la dernière série si elle existe
            if current_series > 0:
                series.append(current_series)

            # Si aucune série n'est trouvée, retourner 0 pour le nombre et None pour la moyenne et l'écart-type
            if not series:
                return 0, None, None

            return len(series), np.mean(series), np.std(series) if len(series) > 1 else 0

        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']
            subset = dataframe[(dataframe['Observation'] == observation) &
                              (dataframe['Year'] == year)].copy()

            # Analyse pour chaque chiffre (0-4)
            for digit in range(5):
                # Dictionnaire pour stocker les statistiques par source
                source_stats = {}
                
                # Calculer les statistiques pour chaque source
                for source in subset['Source'].unique():
                    source_subset = subset[subset['Source'] == source]
                    
                    series_counts = []
                    series_means = []
                    series_stds = []

                    for seq in source_subset['Sequence']:
                        count, mean, std = count_series(seq, digit)
                        series_counts.append(count)
                        if mean is not None:  # N'ajouter la moyenne que si elle existe
                            series_means.append(mean)
                            if std is not None:
                                series_stds.append(std)
                    
                    source_stats[source] = {
                        'counts': series_counts,
                        'means': series_means,
                        'stds': series_stds,
                        'count_mean': np.mean(series_counts) if series_counts else 0,
                        'count_std': np.std(series_counts) if len(series_counts) > 1 else 0,
                        'mean_mean': np.mean(series_means) if series_means else 0,
                        'mean_std': np.std(series_means) if len(series_means) > 1 else 0,
                        'std_mean': np.mean(series_stds) if series_stds else 0,
                        'std_std': np.std(series_stds) if len(series_stds) > 1 else 0
                    }
                
                # Calculer les erreurs entre toutes les paires de sources
                sources = list(source_stats.keys())
                for i in range(len(sources)):
                    for j in range(i+1, len(sources)):
                        source1 = sources[i]
                        source2 = sources[j]
                        
                        count_error = abs(source_stats[source1]['count_mean'] - source_stats[source2]['count_mean'])
                        mean_error = abs(source_stats[source1]['mean_mean'] - source_stats[source2]['mean_mean'])
                        std_error = abs(source_stats[source1]['std_mean'] - source_stats[source2]['std_mean'])
                        
                        # Mettre à jour les statistiques
                        key = (observation, year, source1, source2)
                        if key not in self.stats:
                            self.stats[key] = {}

                        self.stats[key][f"digit_{digit}_series_count_error"] = count_error
                        self.stats[key][f"digit_{digit}_series_mean_error"] = mean_error
                        self.stats[key][f"digit_{digit}_series_std_error"] = std_error
                
                # Créer un graphique pour visualiser les résultats
                fig = go.Figure()
                
                # Préparer les données pour le graphique en barres groupées
                for source in sources:
                    fig.add_trace(go.Bar(
                        name=source,
                        x=['Nombre de séries', 'Taille moyenne', 'Écart-type'],
                        y=[
                            source_stats[source]['count_mean'],
                            source_stats[source]['mean_mean'],
                            source_stats[source]['std_mean']
                        ],
                        error_y=dict(
                            type='data',
                            array=[
                                source_stats[source]['count_std'],
                                source_stats[source]['mean_std'],
                                source_stats[source]['std_std']
                            ],
                            visible=True
                        )
                    ))
                
                # Ajouter les annotations pour les erreurs entre paires
                annotation_y = 0.95
                for i in range(len(sources)):
                    for j in range(i+1, len(sources)):
                        source1 = sources[i]
                        source2 = sources[j]
                        key = (observation, year, source1, source2)
                        
                        if key in self.stats:
                            annotation_text = (
                                f"Erreurs {source1} vs {source2}:\n"
                                f"Nombre: {self.stats[key][f'digit_{digit}_series_count_error']:.3f}\n"
                                f"Taille: {self.stats[key][f'digit_{digit}_series_mean_error']:.3f}\n"
                                f"Std: {self.stats[key][f'digit_{digit}_series_std_error']:.3f}"
                            )
                            
                            fig.add_annotation(
                                x=0.98, y=annotation_y,
                                xref="paper", yref="paper",
                                text=annotation_text,
                                showarrow=False,
                                align="right",
                                font=dict(size=10),
                                bgcolor="white",
                                bordercolor="black",
                                borderwidth=1,
                                borderpad=4,
                                opacity=0.8
                            )
                            annotation_y -= 0.15  # Décaler pour la prochaine annotation
                
                fig.update_layout(
                    title=f"Statistiques des séries du chiffre {digit} pour {observation} en {year}",
                    barmode='group',
                    height=800,
                    width=1200,
                    margin=dict(t=100, b=100, l=50, r=50),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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
    
    # Démonstration de l'analyse des séries dans les séquences
    validator.sequence_series_analysis(df)
    
    # Afficher un aperçu des statistiques collectées
    print("\nStatistiques des séries:")
    for key, value in validator.stats.items():
        if len(key) == 4:  # Format (observation, year, source1, source2)
            print(f"\n{key[0]} - {key[1]}, {key[2]} vs {key[3]}:")
            for stat_name, stat_value in value.items():
                print(f"  {stat_name}: {stat_value:.3f}")
