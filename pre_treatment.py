import re
import os
import pandas as pd
from sequences import terminal_fate



def extract_sequences_to_df(input_file):
    """
    Extract all sequences from the input file and return them as a DataFrame.
    """
    with open(input_file, 'r') as f:
        content = f.read()
    pattern = r'((?:.|\n)*?)#\s*\(\d+\)'
    blocks = re.findall(pattern, content)
    print(f"Nombre de blocs trouvés : {len(blocks)}")

    type_dict = {1:"LARGE",2:"MEDIUM"}
    year_dict = {95:"Y1",96:"Y2",97:"Y3",98:"Y4",99:"Y5",}
    sequence_data = []
    for block in blocks:
        series_pattern = re.compile(r'\d+(?:\s+\d+){6}')
        series_list = series_pattern.findall(block)

        if not series_list:
            continue

        first_series = series_list[0].split()
        print(series_list[0])
        year = first_series[1]
        type_val = first_series[4]
        sequence = ""
        for series in series_list:
            values = series.split()
            sequence += values[6]
            
        sequence_data_line = {
            'Observation': type_dict.get(int(type_val), "UNKNOWN"),
            'Year': year_dict.get(int(year), "UNKNOWN"), 
            'Sequence': sequence,
            'Terminal Fate': terminal_fate(int(year),type_dict.get(int(type_val),"UNKNOWN")),  # Ajout de la colonne "Terminal Fate"
        }
        sequence_data.append(sequence_data_line)
    
    df = pd.DataFrame(sequence_data)
    # Filtrer les séquences de longueur inférieure à 4
    df_filtered = df[df['Sequence'].str.len() >= 4]
    
    # Afficher le nombre de séquences filtrées
    filtered_count = len(df) - len(df_filtered)
    if filtered_count > 0:
        print(f"Suppression de {filtered_count} séquences trop courtes (longueur < 4)")
        
    return df_filtered

def process_all_seq_files(directory='data'):
    """
    Traite tous les fichiers .seq dans le répertoire spécifié et
    combine toutes les séquences extraites dans un seul fichier CSV.
    """
    seq_files = [f for f in os.listdir(directory) if f.endswith('.seq')]
    
    if not seq_files:
        print("Aucun fichier .seq trouvé dans le répertoire.")
        return
    
    total_sequences = 0
    processed_files = 0
    all_sequences_df = pd.DataFrame()
    
    # Traiter chaque fichier .seq
    for seq_file in seq_files:
        input_file = os.path.join(directory, seq_file)
        
        print(f"\nTraitement du fichier: {seq_file}")
        try:
            df_sequences = extract_sequences_to_df(input_file)
            all_sequences_df = pd.concat([all_sequences_df, df_sequences])
            
            num_sequences = len(df_sequences)
            total_sequences += num_sequences
            processed_files += 1
            print(f"Extraction de {num_sequences} séquences de {seq_file}")
        except Exception as e:
            print(f"Erreur lors du traitement de {seq_file}: {str(e)}")
    

    
    # Sauvegarder toutes les séquences dans un seul fichier CSV
    output_file = "data/all_sequences.csv"
    all_sequences_df.to_csv(output_file, index=False)
    
    print(f"\nRésumé: {total_sequences} séquences extraites de {processed_files} fichiers.")
    print(f"Toutes les séquences ont été sauvegardées dans {output_file}")
    
    return total_sequences

if __name__ == "__main__":
    # Traiter tous les fichiers .seq dans le répertoire courant
    process_all_seq_files()
