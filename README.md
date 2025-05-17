# README - Projet de Modélisation de Séquences de Pommier avec Données Originales

## Introduction

Ce projet s\'inscrit dans la continuité des travaux réalisés avec le dépôt "Transformer Pommiers". L\'objectif principal ici est de reprendre la modélisation par transformeur, mais en utilisant cette fois-ci les **séquences de développement réelles des pommiers**. Ces séquences réelles sont celles qui ont initialement servi à créer le modèle de Markov, dont les données générées ont ensuite été utilisées comme source pour le dépôt "Transformer Pommiers" susmentionné.

Ce dépôt offre la flexibilité d\'entraîner un modèle transformeur :
-   Soit **à partir de zéro** en utilisant ces données originales.
-   Soit en effectuant un **fine-tuning** d\'un modèle pré-entraîné (bien que la logique de fine-tuning ne soit pas explicitement détaillée ici, la structure le permettrait).

L\'un des intérêts majeurs est de pouvoir **comparer les performances et la qualité des séquences générées** par un transformeur entraîné sur les données réelles par rapport à :
1.  Aux données réelles elles-mêmes (pour évaluer la fidélité du modèle).
2.  Aux données générées par le modèle de Markov (présentes dans `markov_python_generated_dataset10000.csv`).
3.  Aux données qui seraient générées par un transformeur ent.raîné sur les données issues du modèle de Markov (présentes dans `generated_dataset.csv`)

Le code fourni permet de prétraiter ces données de séquences originales, d\'entraîner les modèles transformeurs, de générer de nouvelles séquences, et de valider leur qualité.

## Structure du Dépôt

Le dépôt est organisé comme suit :

-   `data/`: Contient les fichiers de données brutes (`.seq`) et les fichiers CSV prétraités.
    -   `all_sequences.csv`: Fichier CSV combinant toutes les séquences extraites des fichiers `.seq`.
-   `experiments/`: Contient les résultats des expériences d'entraînement du modèle transformeur (configurations, poids du modèle).
-   `wandb/`: Contient les logs de suivi des expériences avec Weights & Biases (si utilisé).
-   Fichiers Python (`.py`):
    -   `pre_treatment.py`: Scripts pour lire les fichiers `.seq`, extraire les séquences et les sauvegarder en format CSV.
    -   `enums.py`: Définition des énumérations utilisées dans le projet (par exemple, `Observation`).
    -   `PommierDataset.py`: Définition de la classe `Dataset` pour charger les données pour l'entraînement du transformeur.
    -   `transformer.py`: Implémentation du modèle transformeur.
    -   `pipeline.py`: Fonctions pour l'entraînement, la génération de séquences avec le transformeur.
    -   `main.py`: Script principal pour exécuter l'entraînement et la génération avec le transformeur.
    -   `EarlyStopping.py`: Implémentation d'un mécanisme d'arrêt anticipé pour l'entraînement.
    -   `embedding3d.py`: Outils pour visualiser les embeddings du transformeur en 3D.
    -   `SimpleValidator.py`: Classe pour valider et comparer statistiquement les jeux de données de séquences.
    -   `ValidationError.py`: Classe d'exception personnalisée pour les erreurs de validation.
-   `generated_dataset.csv`: Jeu de données généré par le transformeur du dépôt "Transformer Pommiers" (utilisé ici pour comparaison).
-   `markov_python_generated_dataset10000.csv`: Jeu de données généré par un modèle de Markov (pour comparaison).
-   `README.md`: Ce fichier.

## Fonctionnement Général

### 1. Prétraitement des Données (`pre_treatment.py`)

Le script `pre_treatment.py` est responsable de la lecture des fichiers de séquences originaux (généralement au format `.seq`).

-   **Extraction des séquences**:
    -   Les fichiers `.seq` contiennent des blocs d'informations. Chaque bloc représente les observations d'un rameau sur plusieurs années.
    -   Le script parse ces blocs pour extraire des séries de chiffres. Une série typique est une ligne de 7 nombres. Le 7ème nombre de chaque série est interprété comme un "digit" de la séquence (0, 1, 2, 3, 4).
    -   La séquence est formée en concaténant ces digits. Par exemple, si les 7èmes nombres sont `1`, `1`, `2`, `0`, la séquence sera `"1120"`.
    -   Chaque séquence est associée à un type d'observation initial (par exemple, `LARGE`, `MEDIUM`) et une année de début (`Y1` à `Y5`). Ces informations sont extraites des premières séries de chaque bloc.
    -   Les séquences dont la longueur est inférieure à 4 sont filtrées et supprimées.
-   **Calcul du "Terminal Fate"**:
    -   Pour chaque séquence extraite, un "Terminal Fate" (destin terminal) est calculé.
    -   Le destin terminal est une prédiction du type d'organe qui terminera la séquence, basée sur l'année et le type d'observation initial.
    -   Cela est fait en utilisant une distribution de probabilités prédéfinie dans la classe `TerminalFate`. La fonction `terminal_fate` échantillonne un destin à partir de cette distribution.
-   **Sauvegarde en CSV**:
    -   Toutes les séquences extraites (avec leur type, année, et destin terminal) de tous les fichiers `.seq` d\'un répertoire sont combinées en un unique fichier CSV, typiquement `data/all_sequences.csv`. Ce fichier sert de base pour l\'entraînement et la validation.

### 2. Préparation des Données pour l'Entraînement du Transformeur

Cette section décrit comment les données séquentielles, initialement prétraitées par des scripts comme ceux dans `pre_treatment.py` et stockées (par exemple, dans `data/all_sequences.csv`), sont spécifiquement préparées par la classe `PommierDatasetDecoderOnly` (définie dans `PommierDataset.py`) pour l'entraînement d'un modèle Transformeur de type décodeur-seul.

Les étapes suivantes sont effectuées au sein de `PommierDatasetDecoderOnly`:

-   **1. Chargement et Filtrage Initial des Données :**
    -   Les séquences sont chargées à partir du fichier CSV spécifié (`dataset_path`).
    -   Un filtrage strict est appliqué : seules les lignes où l'attribut "Observation" est égal à "MEDIUM" ET où l'attribut "Year" est 'Y4' ou 'Y5' sont conservées pour la suite du traitement.

-   **2. Tokenisation Détaillée des Séquences (`tokenize_row`) :**
    -   Chaque séquence retenue est décomposée en une liste de tokens.
    -   Cette tokenisation suit des règles précises :
        -   Un vocabulaire de base de tokens acceptés est défini : `["LARGE", "MEDIUM", "SMALL", "DORMANT", "FLORAL", "Y1", "Y2", "Y3", "Y4", "Y5"]`. Les éléments de la séquence correspondant à ces tokens sont conservés tels quels.
        -   Si un élément de la séquence est une chaîne de caractères composée uniquement de chiffres (ex: "123"), il est segmenté en tokens individuels pour chaque chiffre (ex: `['1', '2', '3']`).
        -   Tout autre type d'élément est ignoré.
    -   Par exemple, une ligne de données brute pourrait être transformée en une liste de tokens comme `['Y4', 'MEDIUM', '1', '2', '3']`.

-   **3. Construction du Vocabulaire et Conversion en IDs Numériques (`build_vocab`) :**
    -   Si un dictionnaire `token_to_id` n'est pas fourni lors de l'initialisation, un nouveau vocabulaire est construit à partir de l'ensemble des tokens uniques issus de toutes les séquences tokenisées.
    -   Deux tokens spéciaux sont systématiquement ajoutés à ce vocabulaire :
        -   `<PAD>` : Token de padding, utilisé pour égaliser la longueur des séquences au sein d\'un batch. Son ID est généralement 0.
        -   `<SOS>` : Token spécial "Start of Sequence" (Début de Séquence). Son rôle précis dans la formation des séquences d\'entrée est décrit ci-dessous.
    -   *Note importante :* D\'après l\'analyse du code disponible et des informations issues de notre conversation, un token `<EOS>` (End of Sequence / Fin de Séquence) n'est pas explicitement ajouté au vocabulaire ni aux séquences par la logique de `PommierDatasetDecoderOnly` à ce stade.
    -   Après la construction du vocabulaire, chaque token dans les listes de tokens est remplacé par son ID numérique correspondant.

-   **4. Création des Séquences d'Entrée et Cible pour le Modèle Décodeur-Seul (`__getitem__`) :**
    -   L'objectif d'un modèle Transformeur décodeur-seul est de prédire le prochain token d'une séquence. Pour cela, des paires de séquences (entrée, cible) sont générées à partir de chaque séquence d'IDs.
    -   La logique spécifique implémentée dans `PommierDatasetDecoderOnly` est la suivante :
        1.  Soit une séquence de tokens originale, par exemple `S_orig_tokens = ['Y4', 'MEDIUM', '1', '2', '3']`. Après conversion en IDs numériques (où `id_TOKEN` représente l'ID du TOKEN et `id_SOS` l'ID du token `<SOS>`), cela donne `S_orig_ids = [id_Y4, id_MEDIUM, id_1, id_2, id_3]`.
        2.  L'ID du token `<SOS>` est inséré *après les deux premiers tokens* de `S_orig_ids`. La séquence d'IDs modifiée, `S_mod_ids`, devient alors `[id_Y4, id_MEDIUM, id_SOS, id_1, id_2, id_3]`. Cela correspond à une séquence de tokens `S_mod_tokens = ['Y4', 'MEDIUM', '<SOS>', '1', '2', '3']`.
        3.  À partir de `S_mod_ids`, les séquences d'entrée et cible sont formées :
            -   `input_sequence_ids = S_mod_ids[:-1]`, ce qui donne `[id_Y4, id_MEDIUM, id_SOS, id_1, id_2]`.
                -   La séquence de tokens correspondante donnée en entrée au modèle est : `['Y4', 'MEDIUM', '<SOS>', '1', '2']`.
            -   `target_sequence_ids = S_mod_ids[1:]`, ce qui donne `[id_MEDIUM, id_SOS, id_1, id_2, id_3]`.
                -   La séquence de tokens correspondante que le modèle doit apprendre à prédire est : `['MEDIUM', '<SOS>', '1', '2', '3']`.

        -   **Impact du Découpage (Trimming) dans la Boucle d'Entraînement (`pipeline.py`) :**
            -   Il est crucial de noter que, lors de la phase d'entraînement (comme vu dans `pipeline.py`), les logits (sorties brutes du modèle) et les séquences cibles sont découpées avant le calcul de la perte : `logits_trim = logits[:, 2:, :]` et `targets_trim = target_seq[:, 2:]`.
            -   Cela signifie que les deux premiers éléments de la `target_sequence_ids` (c'est-à-dire `id_MEDIUM` et `id_SOS` dans notre exemple) ne sont **pas** utilisés pour le calcul de la perte. De même, les prédictions correspondantes du modèle pour ces deux premiers tokens cibles sont ignorées.
            -   Par conséquent, l'apprentissage effectif commence par la prédiction du token qui suit le token `<SOS>` dans la séquence `S_mod_tokens`.
            -   En reprenant notre exemple `S_mod_tokens = ['Y4', 'MEDIUM', '<SOS>', '1', '2', '3']`:
                -   L'entrée effective pour la première prédiction pertinente est `['Y4', 'MEDIUM', '<SOS>']` (les trois premiers tokens de `input_sequence_tokens`).
                -   Le modèle apprend à prédire `'1'` (le quatrième token de `S_mod_tokens`, qui est le premier token de `targets_trim`).
                -   Ensuite, avec l'entrée `['Y4', 'MEDIUM', '<SOS>', '1']`, il apprend à prédire `'2'`.
                -   Et avec `['Y4', 'MEDIUM', '<SOS>', '1', '2']`, il apprend à prédire `'3'`.

-   **5. Collation et Padding des Batches (`collate_fn_decoder_only`) :**
    -   Lorsque les données sont chargées par un `DataLoader`, la fonction `collate_fn_decoder_only` est utilisée.
    -   Son rôle est de regrouper plusieurs paires `(input_sequence_ids, target_sequence_ids)` en un batch.
    -   Comme les séquences peuvent avoir des longueurs différentes, elles sont paddées (remplies) avec l'ID du token `<PAD>` (valeur 0) pour atteindre une longueur uniforme au sein du batch, généralement la longueur de la plus longue séquence du batch.

-   **6. Calcul des Poids pour Échantillonnage Stratifié (`WeightedRandomSampler`) :**
    -   Pour potentiellement atténuer les déséquilibres entre différentes catégories de séquences (basées sur les combinaisons "Observation" et "Year"), `PommierDatasetDecoderOnly` calcule des poids.
    -   Chaque séquence se voit assigner un poids qui est l'inverse de la fréquence du groupe auquel elle appartient (`poids = 1.0 / nombre_de_séquences_dans_le_groupe`).
    -   Ces poids peuvent ensuite être utilisés par un `WeightedRandomSampler` dans le `DataLoader` (configuré dans des scripts comme `pipeline.py` ou `main.py`) pour sur-échantillonner les séquences des groupes moins représentés durant la phase d'entraînement.

Cette préparation minutieuse des données est cruciale pour l'entraînement efficace du modèle Transformeur.

### 3. Validation des Séquences Générées (`SimpleValidator.py`)

La classe `SimpleValidator` est cruciale pour évaluer la qualité des séquences générées par les modèles en les comparant aux séquences du jeu de données original. Elle prend en entrée les chemins vers les fichiers CSV des jeux de données à comparer (par exemple, les données originales et les données générées).

-   **Initialisation et Filtrage (`__init__`, `filter_and_include_all_if_present`):**
    -   Lors de l'initialisation, on peut spécifier si les graphiques de validation doivent être affichés (`show_plots`).
    -   La méthode `filter_and_include_all_if_present` est une étape de préparation importante. Elle s'assure que la comparaison se fait sur des bases équitables :
        1.  Elle identifie les paires (`Observation`, `Année`) qui sont présentes dans **tous** les jeux de données fournis.
        2.  Pour ces paires communes, elle conserve **toutes** les séquences associées de chaque jeu de données. Il n'y a pas d'égalisation du nombre de séquences entre les sources à ce stade pour ces paires communes.
        -   L'objectif est de comparer les distributions complètes pour les conditions (Observation, Année) où tous les modèles ont pu produire des données.

-   **Méthodes de Validation Détaillées:**

    1.  **`sequence_length_validation(dataframe)`**:
        -   **Objectif**: Comparer la longueur moyenne et l'écart-type des séquences.
        -   **Fonctionnement**:
            -   Pour chaque paire (`Observation`, `Année`) commune, elle calcule la longueur de chaque séquence.
            -   Elle groupe ensuite par `Source` (nom du jeu de données) et calcule la moyenne et l'écart-type des longueurs.
            -   **Visualisation**: Un graphique à points (scatter plot) est généré pour chaque paire (`Observation`, `Année`). Chaque point représente une `Source`, avec sa longueur moyenne en ordonnée et des barres d'erreur indiquant l'écart-type. Cela permet de voir si les longueurs moyennes sont similaires et si la variabilité (écart-type) est comparable.

    2.  **`sequence_length_distribution_validation(dataframe)`**:
        -   **Objectif**: Comparer la distribution complète des longueurs de séquences.
        -   **Fonctionnement**:
            -   Pour chaque paire (`Observation`, `Année`) commune:
                -   Calcule la longueur de chaque séquence.
                -   Génère des histogrammes de densité de probabilité des longueurs pour chaque `Source`.
                -   Superpose des estimations de densité par noyau (KDE) pour lisser les distributions.
                -   Calcule la **distance de Jensen-Shannon (JS)** entre les distributions KDE de chaque paire de sources. La distance JS mesure la similarité entre deux distributions de probabilités (une valeur plus faible indique une plus grande similarité).
            -   **Visualisation**: Un graphique par paire (`Observation`, `Année`) montrant les histogrammes et les courbes KDE superposées. Les distances JS calculées sont stockées et peuvent être affichées.

    3.  **`sequence_digit_stats(dataframe, observation_filter=None, year_filter=None)`**:
        -   **Objectif**: Analyser la fréquence de chaque digit (0 à 4) dans les séquences.
        -   **Fonctionnement**:
            -   Pour chaque paire (`Observation`, `Année`) (éventuellement filtrée):
                -   Pour chaque digit (0, 1, 2, 3, 4):
                    -   Pour chaque `Source`, elle compte le nombre d'occurrences du digit dans chaque séquence.
                    -   Elle calcule ensuite la moyenne et l'écart-type de ces comptes pour cette `Source`.
                -   Calcule les erreurs absolues (différences) des moyennes et des écarts-types entre les paires de `Source`.
            -   **Visualisation**: Pour chaque digit et chaque paire (`Observation`, `Année`), un diagramme à barres est généré. Chaque barre représente une `Source`, montrant le nombre moyen d'occurrences du digit, avec des barres d'erreur pour l'écart-type. Les erreurs calculées entre les sources sont souvent affichées sous forme d'annotations.

    4.  **`sequence_series_analysis(dataframe, observation_filter=None, year_filter=None)`**:
        -   **Objectif**: Analyser les "séries" de digits consécutifs. Une série est une suite ininterrompue du même digit (par exemple, `111` dans `01112`).
        -   **Fonctionnement**:
            -   Pour chaque paire (`Observation`, `Année`) (éventuellement filtrée):
                -   Pour chaque digit (0 à 4):
                    -   Pour chaque `Source` et pour chaque séquence de cette source:
                        -   La méthode `count_series` est appelée. Elle identifie toutes les séries du digit cible dans la séquence.
                        -   Elle retourne : le nombre de séries trouvées, la longueur moyenne de ces séries, et l'écart-type des longueurs de ces séries.
                    -   Ensuite, pour chaque `Source`, on agrège ces statistiques sur toutes ses séquences :
                        -   Moyenne du nombre de séries par séquence.
                        -   Moyenne des longueurs moyennes des séries.
                        -   Moyenne des écarts-types des longueurs des séries.
                        -   Des écarts-types correspondants pour ces moyennes agrégées sont aussi calculés.
                -   Calcule les erreurs absolues pour ces statistiques agrégées (nombre moyen de séries, taille moyenne des séries, etc.) entre les paires de `Source`.
            -   **Visualisation**: Pour chaque digit et chaque paire (`Observation`, `Année`), un diagramme à barres groupées est généré. Il montre, pour chaque `Source`, les trois métriques agrégées (nombre moyen de séries, taille moyenne des séries, écart-type moyen des tailles de séries), avec des barres d'erreur. Les erreurs calculées sont souvent affichées en annotations.

-   **Résumé des Statistiques (`print_stats_summary`)**:
    -   Affiche un résumé des statistiques d'erreur (distances JS, erreurs sur les stats des digits et des séries) collectées entre les différentes sources pour les paires (`Observation`, `Année`).

### 4. Entraînement et Génération (Exemple avec le Transformeur - `main.py`)

Le fichier `main.py` orchestre l'entraînement du modèle transformeur et la génération de séquences.

-   **Configuration**: Définit les hyperparamètres du modèle (dimension d'embedding, nombre de têtes d'attention, etc.), les paramètres d'entraînement (batch size, nombre d'époques, learning rate), et les chemins des fichiers.
-   **Chargement des données**: Utilise `PommierDataset` et `DataLoader` de PyTorch.
-   **Entraînement**:
    -   Initialise le modèle transformeur, l'optimiseur, et la fonction de perte (CrossEntropyLoss).
    -   Boucle d'entraînement sur plusieurs époques.
    -   Utilise `EarlyStopping` pour arrêter l'entraînement si la perte sur le jeu de validation ne s'améliore plus.
    -   Sauvegarde les poids du meilleur modèle.
-   **Génération**:
    -   Charge les poids du meilleur modèle entraîné.
    -   Appelle la fonction `generate_sequences` (de `pipeline.py`) pour produire un nouveau jeu de données de séquences.
    -   Les séquences générées sont sauvegardées dans un fichier CSV.

## Comment Utiliser

1.  **Préparer l'environnement**: 
    -   Assurez-vous d'avoir Python installé sur votre système.
    -   Ce projet utilise `uv` comme gestionnaire de paquets et d'environnement virtuel. Si vous ne l'avez pas, installez `uv` en suivant les instructions sur [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).
    -   Une fois `uv` installé, synchronisez l'environnement et installez toutes les dépendances nécessaires (PyTorch, Pandas, NumPy, Plotly, Scikit-learn, UMAP, etc.) en exécutant la commande suivante à la racine du projet :
    ```bash
    uv sync
    ```
    -   Cela créera un environnement virtuel (s'il n'existe pas) et installera les paquets listés dans `pyproject.toml` et `uv.lock`.
2.  **Prétraitement**:
    -   Placez vos fichiers `.seq` dans un répertoire (par exemple, `data/`).
    -   Exécutez `pre_treatment.py` (en ajustant potentiellement le répertoire d'entrée dans le script) pour générer `data/all_sequences.csv`.
    ```bash
    python pre_treatment.py
    ```
3.  **Entraînement du Transformeur (Optionnel)**:
    -   Modifiez les configurations dans `main.py` si nécessaire.
    -   Exécutez `main.py` pour entraîner un nouveau modèle transformeur et générer des séquences.
    ```bash
    python main.py
    ```
    -   Les poids du modèle et les séquences générées seront sauvegardés dans le répertoire `experiments/` et à la racine respectivement.
4.  **Validation**:
    -   Assurez-vous d'avoir au moins deux fichiers CSV de séquences à comparer (par exemple, `data/all_sequences.csv` et `generated_dataset.csv` ou `markov_python_generated_dataset10000.csv`).
    -   Modifiez les `dataset_paths` et `dataset_names` dans la fonction `main` de `SimpleValidator.py` pour pointer vers vos fichiers.
    -   Exécutez `SimpleValidator.py`:
    ```bash
    python SimpleValidator.py
    ```
    -   Les graphiques de validation s'afficheront dans votre navigateur (si `show_plots=True`) et un résumé des statistiques sera imprimé dans la console.


