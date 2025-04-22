import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'browser'
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import webbrowser
webbrowser.register('firefox', None, webbrowser.BackgroundBrowser('/usr/bin/firefox'))


class SimpleValidator:
    def __init__(self) -> None:
        self.stats = {}
        self.show = True

    def filter_and_include_all_if_present(self, csv_paths, dataset_names):
        """
        Filtre les données pour ne garder que les paires (Observation, Year)
        présentes dans TOUS les datasets, et inclut TOUTES les séquences
        pour ces paires sans égalisation.

        Args:
            csv_paths (list): Liste des chemins vers les fichiers CSV.
            dataset_names (list): Liste des noms correspondants aux datasets.

        Returns:
            pd.DataFrame: DataFrame filtré contenant toutes les séquences des
                          paires communes, avec une colonne 'Source'.
        """
        dataframes = [pd.read_csv(path, dtype={"Sequence": str}) for path in csv_paths]

        for df in dataframes:
            # Assurer les types de colonnes
            df['Observation'] = df['Observation'].astype(str)
            df['Year'] = df['Year'].astype(str)
            # Filtrer les séquences potentiellement vides ou invalides au début si nécessaire
            # df = df[df["Sequence"].notna() & (df["Sequence"] != "")] # Exemple

        # Trouver les paires (Observation, Year) uniques dans chaque dataframe
        pairs_per_df = [
            set(df[['Observation', 'Year']].drop_duplicates().apply(tuple, axis=1))
            for df in dataframes
        ]

        # Trouver l'intersection : les paires présentes dans TOUS les dataframes
        common_pairs = set.intersection(*pairs_per_df)

        # Initialiser le DataFrame final
        filtered_df = pd.DataFrame()

        # Itérer sur les paires communes
        for obs, year in common_pairs:
            # Vérifier si cette paire a au moins une séquence dans CHAQUE dataframe
            # (Cette vérification est implicitement faite par l'intersection,
            # mais on pourrait la rendre explicite si nécessaire pour la clarté
            # ou si des filtres supplémentaires étaient appliqués avant l'intersection)

            # Récupérer TOUTES les lignes pour cette paire de chaque dataframe
            subsets_to_concat = []
            all_present = True # Flag pour vérifier la présence dans tous les DFs après filtrage potentiel
            for df, name in zip(dataframes, dataset_names):
                subset = df[(df['Observation'] == obs) & (df['Year'] == year)].copy()
                # Vérification supplémentaire : s'assurer qu'il y a bien des données
                # pour cette paire DANS CE dataframe spécifique.
                if subset.empty:
                    all_present = False
                    break # Si un DF n'a rien pour cette paire, on l'ignore complètement
                subset['Source'] = name
                subsets_to_concat.append(subset)

            # Si la paire est bien présente et non vide dans tous les dataframes
            if all_present:
                # Concaténer tous les sous-ensembles pour cette paire
                filtered_df = pd.concat([filtered_df] + subsets_to_concat, ignore_index=True)

        return filtered_df

    # ... (le reste des méthodes : sequence_length_validation, etc. restent inchangées) ...
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

            # Gérer les cas où std pourrait être NaN (si un groupe a une seule séquence)
            stats['std'] = stats['std'].fillna(0)

            # Calculer les limites y en tenant compte des std potentiellement nulles
            max_std = stats['std'].max()
            y_max = stats['mean'].max() + (10 * max_std if max_std > 0 else 10) # Ajoute un petit espace si std=0
            y_min = stats['mean'].min() - (10 * max_std if max_std > 0 else 10) # Ajoute un petit espace si std=0
            # Assurer que y_min n'est pas négatif si les longueurs sont toujours positives
            y_min = max(0, y_min)


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
        Analyse la distribution des longueurs de séquences en utilisant des histogrammes et la distance de Jensen-Shannon.

        Args:
            dataframe (pd.DataFrame): DataFrame contenant les séquences à analyser
                                      avec une colonne 'Source' pour distinguer l'origine.
        """
        # Filtrer les séquences vides
        dataframe = dataframe[dataframe["Sequence"] != "0"]
        dataframe = dataframe[(dataframe['Observation'] == 'MEDIUM') & (dataframe['Year'].isin(['Y4', 'Y5']))]

        unique_pairs = dataframe[['Observation', 'Year']].drop_duplicates()

        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']
            subset = dataframe[(dataframe['Observation'] == observation) &
                              (dataframe['Year'] == year)].copy()

            # Calculer la longueur de chaque séquence
            subset['Sequence Length'] = subset['Sequence'].apply(len)

            # Séparer les données par source
            sources = subset['Source'].unique()

            # Créer la figure pour l'histogramme
            fig = go.Figure()

            # Variables pour stocker les distributions KDE
            kde_norm_values = {}
            x_grids = {} # Stocker les grilles x pour chaque source

            # Collecter les longueurs par source
            all_lengths = subset['Sequence Length'].values
            if len(all_lengths) == 0: continue # Passer si pas de données pour cette paire
            global_min_len = min(all_lengths)
            global_max_len = max(all_lengths)

            for source in sources:
                # Extraire les longueurs pour cette source
                lengths = subset[subset['Source'] == source]['Sequence Length'].values
                if len(lengths) == 0: continue # Passer si pas de données pour cette source

                # Ajouter l'histogramme pour cette source
                fig.add_trace(go.Histogram(
                    x=lengths,
                    name=source,
                    opacity=0.6,
                    histnorm='probability density'
                ))

                # Calculer le KDE si possible (au moins 2 points et variance > 0)
                if len(lengths) > 1 and np.var(lengths) > 0:
                    try:
                        # Définir une grille x commune basée sur l'étendue globale
                        x_grid = np.linspace(global_min_len, global_max_len, 500) # Grille commune
                        x_grids[source] = x_grid

                        # Calculer le KDE
                        kde = gaussian_kde(lengths, bw_method='scott') # 'scott' est souvent un bon point de départ
                        kde_values = kde(x_grid)

                        # Normaliser pour Jensen-Shannon
                        dx = (global_max_len - global_min_len) / (len(x_grid) - 1) if len(x_grid) > 1 else 1
                        kde_norm = kde_values / np.sum(kde_values * dx)
                        kde_norm_values[source] = kde_norm

                        # Ajouter la courbe KDE
                        fig.add_trace(go.Scatter(
                            x=x_grid,
                            y=kde_norm,
                            mode='lines',
                            name=f'KDE {source}'
                        ))
                    except Exception as e:
                        print(f"Erreur lors du calcul du KDE pour {source} ({observation}, {year}): {e}")
                        # S'assurer que la source n'est pas dans kde_norm_values si le calcul échoue
                        if source in kde_norm_values: del kde_norm_values[source]
                        if source in x_grids: del x_grids[source]

            # Calculer les distances Jensen-Shannon entre les sources ayant un KDE valide
            valid_sources = list(kde_norm_values.keys())
            js_distances_text = [] # Pour affichage

            for i in range(len(valid_sources)):
                for j in range(i+1, len(valid_sources)):
                    source1 = valid_sources[i]
                    source2 = valid_sources[j]

                    # S'assurer que les grilles sont compatibles (elles devraient l'être maintenant)
                    if source1 in x_grids and source2 in x_grids and len(x_grids[source1]) == len(x_grids[source2]):
                        # Calculer la distance JS
                        # Ajouter un petit epsilon pour éviter les log(0) si une densité est nulle
                        epsilon = 1e-10
                        p = kde_norm_values[source1] + epsilon
                        q = kde_norm_values[source2] + epsilon
                        js_distance = jensenshannon(p, q)

                        # Stocker dans les statistiques
                        key = (observation, year, source1, source2)
                        if key not in self.stats:
                            self.stats[key] = {}
                        self.stats[key]["sequence_length_js_distance"] = js_distance
                        js_distances_text.append(f"{source1} vs {source2} JS: {js_distance:.4f}")

                    else:
                         print(f"Impossible de calculer JS pour {source1} vs {source2} ({observation}, {year}) en raison de KDE non calculés ou de grilles incompatibles.")


            # Ajouter une annotation unique pour toutes les distances JS
            if js_distances_text:
                 fig.add_annotation(
                    x=0.98, y=0.98,
                    xref="paper", yref="paper",
                    text="<br>".join(js_distances_text), # Utiliser <br> pour les sauts de ligne dans Plotly
                    showarrow=False,
                    align="right",
                    bgcolor="rgba(255,255,255,0.8)", # Fond légèrement transparent
                    bordercolor="black",
                    borderwidth=1
                 )


            # Mise en page finale
            fig.update_layout(
                title=f"Distribution des longueurs de séquences pour {observation} en {year}",
                xaxis_title="Longueur de séquence",
                yaxis_title="Densité de probabilité",
                barmode='overlay', # Superposer les histogrammes
                legend_title_text='Source',
                height=600,
                width=1000
            )
            fig.update_xaxes(range=[global_min_len, global_max_len]) # Définir l'axe x globalement

            # Afficher la figure
            if self.show:
                fig.show()

    def sequence_digit_stats(self, dataframe):
        """
        Analyse les statistiques des chiffres dans les séquences et compare entre les datasets.

        Args:
            dataframe (pd.DataFrame): DataFrame contenant les séquences à analyser
                                      avec une colonne 'Source' pour distinguer l'origine.
        """
        # Filtrer les séquences vides
        dataframe = dataframe[dataframe["Sequence"] != "0"]
        dataframe = dataframe[(dataframe['Observation'] == 'MEDIUM') & (dataframe['Year'].isin(['Y4', 'Y5']))]

        unique_pairs = dataframe[['Observation', 'Year']].drop_duplicates()

        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']
            subset = dataframe[(dataframe['Observation'] == observation) &
                              (dataframe['Year'] == year)].copy()

            sources = subset['Source'].unique()
            if len(sources) < 2: continue # Pas de comparaison possible si moins de 2 sources

            # Pour chaque chiffre (0-4), créer un graphique dédié
            for digit in range(5):
                # Calculer les statistiques pour chaque source
                stats_data = []

                for source in sources:
                    source_data = subset[subset['Source'] == source]
                    if source_data.empty: continue # Ignorer si pas de données pour cette source/paire

                    # Compter les occurrences du chiffre dans chaque séquence
                    # Assurer que seq est une chaîne avant de compter
                    counts = [str(seq).count(str(digit)) for seq in source_data['Sequence'] if pd.notna(seq)]
                    if not counts: # Si aucune séquence valide n'est trouvée
                         mean_count = 0
                         std_count = 0
                    else:
                         mean_count = np.mean(counts)
                         std_count = np.std(counts) # std de 0 si len(counts) == 1

                    stats_data.append({
                        'Source': source,
                        'Mean': mean_count,
                        'Std': std_count
                    })

                # Si on n'a pas pu calculer de stats pour au moins 2 sources, passer au chiffre suivant
                if len(stats_data) < 2: continue

                # Créer un DataFrame pour le graphique
                digit_stats_df = pd.DataFrame(stats_data)

                # Calculer les erreurs entre les sources
                errors_text = []
                for i in range(len(sources)):
                    for j in range(i+1, len(sources)):
                        source1 = sources[i]
                        source2 = sources[j]

                        # Trouver les stats correspondantes dans digit_stats_df
                        stats1_row = digit_stats_df[digit_stats_df['Source'] == source1]
                        stats2_row = digit_stats_df[digit_stats_df['Source'] == source2]

                        # S'assurer que les deux sources ont des données
                        if stats1_row.empty or stats2_row.empty: continue

                        stats1 = stats1_row.iloc[0]
                        stats2 = stats2_row.iloc[0]

                        mean_error = abs(stats1['Mean'] - stats2['Mean'])
                        std_error = abs(stats1['Std'] - stats2['Std'])

                        # Stocker les erreurs dans le dictionnaire self.stats
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


                # Créer le graphique à barres pour ce chiffre
                fig = px.bar(
                    digit_stats_df,
                    x='Source',
                    y='Mean',
                    error_y='Std',
                    title=f'Digit {digit} Occurrence Stats - {observation} in {year}',
                    labels={'Mean': f'Avg Occurrences of Digit {digit}', 'Std': 'Standard Deviation'},
                    color='Source'
                )

                # Ajouter une annotation unique pour les erreurs
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

                # Mise en page du graphique
                fig.update_layout(
                    xaxis_title="Source",
                    yaxis_title=f"Average occurrences of digit {digit}",
                    height=600,
                    width=800,
                    margin=dict(l=50, r=250, t=80, b=50) # Augmenter la marge droite pour l'annotation
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
        # Filtrer les séquences vides ou non valides
        dataframe = dataframe[dataframe["Sequence"].notna() & (dataframe["Sequence"] != "0")]
        # S'assurer que Sequence est de type string pour l'itération
        dataframe['Sequence'] = dataframe['Sequence'].astype(str)

        # Filtrage spécifique si nécessaire (décommenter et adapter si besoin)
        # dataframe = dataframe[(dataframe['Observation'] == 'MEDIUM') & (dataframe['Year'].isin(['Y4', 'Y5']))]

        unique_pairs = dataframe[['Observation', 'Year']].drop_duplicates()

        def count_series(sequence, digit):
            """Compte le nombre de séries d'un chiffre donné et leur taille moyenne."""
            series_lengths = []
            current_series_length = 0
            digit_char = str(digit) # Convertir le chiffre en caractère une seule fois

            for char in sequence:
                if char == digit_char:
                    current_series_length += 1
                else:
                    if current_series_length > 0:
                        series_lengths.append(current_series_length)
                    current_series_length = 0 # Réinitialiser si le caractère change

            # Ajouter la dernière série si la séquence se termine par le chiffre recherché
            if current_series_length > 0:
                series_lengths.append(current_series_length)

            # Calculer les statistiques sur les longueurs de séries trouvées
            num_series = len(series_lengths)
            if num_series == 0:
                mean_len = 0
                std_len = 0
            else:
                mean_len = np.mean(series_lengths)
                # Calculer std seulement si plus d'une série existe
                std_len = np.std(series_lengths) if num_series > 1 else 0

            return num_series, mean_len, std_len

        for _, row in unique_pairs.iterrows():
            observation = row['Observation']
            year = row['Year']
            subset = dataframe[(dataframe['Observation'] == observation) &
                              (dataframe['Year'] == year)].copy()

            sources = subset['Source'].unique()
            if len(sources) < 2: continue # Besoin d'au moins deux sources pour comparer

            # Analyse pour chaque chiffre (0-4)
            for digit in range(5):
                # Dictionnaire pour stocker les statistiques agrégées par source
                source_aggregated_stats = {}

                # Calculer les statistiques pour chaque source
                valid_sources_for_digit = []
                for source in sources:
                    source_subset = subset[subset['Source'] == source]
                    if source_subset.empty: continue

                    # Listes pour stocker les résultats par séquence pour cette source
                    all_num_series = []
                    all_mean_lengths = []
                    all_std_lengths = []

                    for seq in source_subset['Sequence']:
                        # Assurer que seq est une chaîne non vide
                        if isinstance(seq, str) and seq:
                            num, mean_len, std_len = count_series(seq, digit)
                            all_num_series.append(num)
                            # Ajouter mean et std seulement si au moins une série a été trouvée (mean/std > 0)
                            if num > 0:
                                all_mean_lengths.append(mean_len)
                                all_std_lengths.append(std_len)

                    # Calculer les statistiques agrégées pour la source
                    # Moyenne du nombre de séries par séquence
                    count_mean = np.mean(all_num_series) if all_num_series else 0
                    count_std = np.std(all_num_series) if len(all_num_series) > 1 else 0
                    # Moyenne des tailles moyennes de séries (sur les séquences qui avaient des séries)
                    mean_mean = np.mean(all_mean_lengths) if all_mean_lengths else 0
                    mean_std = np.std(all_mean_lengths) if len(all_mean_lengths) > 1 else 0
                    # Moyenne des écarts-types des tailles de séries (sur les séquences qui avaient des séries)
                    std_mean = np.mean(all_std_lengths) if all_std_lengths else 0
                    std_std = np.std(all_std_lengths) if len(all_std_lengths) > 1 else 0


                    source_aggregated_stats[source] = {
                        'count_mean': count_mean, 'count_std': count_std,
                        'mean_mean': mean_mean, 'mean_std': mean_std,
                        'std_mean': std_mean, 'std_std': std_std
                    }
                    valid_sources_for_digit.append(source)

                # Si moins de deux sources ont des données valides pour ce chiffre, passer au suivant
                if len(valid_sources_for_digit) < 2: continue

                # Calculer les erreurs entre toutes les paires de sources valides
                errors_text = []
                current_sources = valid_sources_for_digit # Utiliser les sources qui ont effectivement des données
                for i in range(len(current_sources)):
                    for j in range(i+1, len(current_sources)):
                        source1 = current_sources[i]
                        source2 = current_sources[j]

                        stats1 = source_aggregated_stats[source1]
                        stats2 = source_aggregated_stats[source2]

                        count_error = abs(stats1['count_mean'] - stats2['count_mean'])
                        mean_error = abs(stats1['mean_mean'] - stats2['mean_mean'])
                        std_error = abs(stats1['std_mean'] - stats2['std_mean'])

                        # Mettre à jour les statistiques globales
                        key = (observation, year, source1, source2)
                        if key not in self.stats: self.stats[key] = {}
                        if f"digit_{digit}_series_errors" not in self.stats[key]: self.stats[key][f"digit_{digit}_series_errors"] = {}

                        self.stats[key][f"digit_{digit}_series_errors"]['count_error'] = count_error
                        self.stats[key][f"digit_{digit}_series_errors"]['mean_error'] = mean_error
                        self.stats[key][f"digit_{digit}_series_errors"]['std_error'] = std_error
                        errors_text.append(f"{source1} vs {source2}: CntErr:{count_error:.2f}, MeanErr:{mean_error:.2f}, StdErr:{std_error:.2f}")


                # Créer un graphique pour visualiser les résultats
                fig = go.Figure()

                # Préparer les données pour le graphique en barres groupées
                plot_data = {'Source': [], 'Metric': [], 'Value': [], 'StdDev': []}
                metrics_map = {
                    'Nombre moyen de séries': ('count_mean', 'count_std'),
                    'Taille moyenne des séries': ('mean_mean', 'mean_std'),
                    'Std moyen des tailles de séries': ('std_mean', 'std_std')
                }

                for source in current_sources:
                    stats = source_aggregated_stats[source]
                    for metric_name, (mean_key, std_key) in metrics_map.items():
                         plot_data['Source'].append(source)
                         plot_data['Metric'].append(metric_name)
                         plot_data['Value'].append(stats[mean_key])
                         plot_data['StdDev'].append(stats[std_key])

                plot_df = pd.DataFrame(plot_data)

                # Créer le graphique groupé
                fig = px.bar(plot_df, x='Metric', y='Value', color='Source',
                             barmode='group', error_y='StdDev',
                             title=f"Statistiques des séries du chiffre {digit} - {observation} {year}",
                             labels={'Value': 'Valeur moyenne', 'Metric': 'Statistique', 'StdDev': 'Écart-type de la moyenne'})


                # Ajouter les annotations pour les erreurs entre paires
                if errors_text:
                    fig.add_annotation(
                        x=1.0, y=1.0, # Positionner en haut à droite
                        xref="paper", yref="paper",
                        text="Erreurs absolues moyennes:<br>" + "<br>".join(errors_text),
                        showarrow=False,
                        align="right",
                        font=dict(size=10),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4
                    )

                fig.update_layout(
                    height=700, # Ajuster la hauteur si nécessaire
                    width=1200,
                    margin=dict(t=100, b=100, l=50, r=300), # Augmenter marge droite pour annotations
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                if self.show:
                    fig.show()


if __name__ == "__main__":
    validator = SimpleValidator()
    validator.show = True
    dataset_paths = [
        "generated_dataset.csv",
        "markov_python_generated_dataset10000.csv",
        'data/all_sequences.csv'
    ]
    dataset_names = [
        "exp trans",
        "markov",
        "original_data"
    ]
    # Utiliser la nouvelle méthode de filtrage/inclusion
    # df = validator.filter_and_equalize_multiple_datasets(dataset_paths, dataset_names) # Ancienne méthode
    df = validator.filter_and_include_all_if_present(dataset_paths, dataset_names) # Nouvelle méthode
    print(f"DataFrame filtré contient {len(df)} lignes.")
    print("Répartition par source:")
    print(df['Source'].value_counts())
    print("\nPaires (Observation, Year) présentes:")
    print(df[['Observation', 'Year']].drop_duplicates())


    # Appeler les méthodes d'analyse (elles restent inchangées)
    validator.sequence_length_validation(df.copy()) # Utiliser .copy() pour éviter les SettingWithCopyWarning

    validator.sequence_length_distribution_validation(df.copy())

    validator.sequence_digit_stats(df.copy())

    validator.sequence_series_analysis(df.copy())

    # Afficher un aperçu des statistiques collectées
    print("\nStatistiques Collectées (Aperçu):")
    count = 0
    for key, value in validator.stats.items():
        if len(key) == 4:  # Format (observation, year, source1, source2)
            print(f"\n{key[0]} - {key[1]}, {key[2]} vs {key[3]}:")
            for stat_name, stat_value in value.items():
                # Gérer les dictionnaires imbriqués (pour les erreurs par chiffre)
                if isinstance(stat_value, dict):
                    print(f"  {stat_name}:")
                    details = []
                    for sub_key, sub_value in stat_value.items():
                         # Essayer de formater en float, sinon afficher tel quel
                         try:
                             details.append(f"{sub_key}: {sub_value:.3f}")
                         except (TypeError, ValueError):
                             details.append(f"{sub_key}: {sub_value}")
                    print(f"    {', '.join(details)}")

                else:
                    # Essayer de formater en float, sinon afficher tel quel
                    try:
                        print(f"  {stat_name}: {stat_value:.3f}")
                    except (TypeError, ValueError):
                        print(f"  {stat_name}: {stat_value}")
            count += 1
            if count >= 5: # Limiter l'aperçu pour ne pas surcharger la sortie
                print("\n[... plus de statistiques ...]")
                break
    if not validator.stats:
        print("Aucune statistique de comparaison n'a été calculée (peut-être pas de paires communes ou pas assez de données).")
