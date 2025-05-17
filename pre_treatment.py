"""
This module provides functions and classes for preprocessing sequence files, extracting data, and calculating terminal fate.
"""

import re
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enums import Observation


TerminalFateData = Dict[Tuple[int, Observation], List[float]]

class TerminalFate:
    """
    Class to deal with terminal fate probabilities
    """
    data: TerminalFateData
    codes: Dict[Observation, int]

    def __init__(self, data: Optional[TerminalFateData] = None):
        if data is not None:
            self.data = data
        else:
            self.data = {
                (1, Observation.LARGE): [0.500, 0.167, 0.000, 0.333],
                (1, Observation.MEDIUM): [0.000, 0.000, 0.000, 1.000],
                (1, Observation.SMALL): [0.100, 0.100, 0.300, 0.500],
                (1, Observation.FLORAL): [0.100, 0.300, 0.600, 0.000],
                (2, Observation.LARGE): [0.246, 0.185, 0.000, 0.569],
                (2, Observation.MEDIUM): [0.016, 0.238, 0.032, 0.714],
                (2, Observation.SMALL): [0.066, 0.067, 0.317, 0.550],
                (2, Observation.FLORAL): [0.317, 0.250, 0.433, 0.000],
                (3, Observation.LARGE): [0.351, 0.106, 0.010, 0.533],
                (3, Observation.MEDIUM): [0.123, 0.148, 0.063, 0.666],
                (3, Observation.SMALL): [0.015, 0.094, 0.453, 0.438],
                (3, Observation.FLORAL): [0.182, 0.249, 0.569, 0.000],
                (4, Observation.LARGE): [0.213, 0.082, 0.000, 0.705],
                (4, Observation.MEDIUM): [0.027, 0.046, 0.016, 0.911],
                (4, Observation.SMALL): [0.000, 0.024, 0.205, 0.771],
                (4, Observation.FLORAL): [0.003, 0.413, 0.584, 0.000],
                (5, Observation.LARGE): [0.100, 0.050, 0.000, 0.850],
                (5, Observation.MEDIUM): [0.000, 0.020, 0.130, 0.850],
                (5, Observation.SMALL): [0.000, 0.000, 0.375, 0.625],
                (5, Observation.FLORAL): [0.008, 0.325, 0.667, 0.000],
                (6, Observation.LARGE): [0.000, 0.100, 0.000, 0.900],
                (6, Observation.MEDIUM): [0.000, 0.050, 0.050, 0.900],
                (6, Observation.SMALL): [0.000, 0.000, 0.350, 0.650],
                (6, Observation.FLORAL): [0.000, 0.200, 0.800, 0.000],
            }
        self.codes = {
            Observation.LARGE: 0,
            Observation.MEDIUM: 1,
            Observation.SMALL: 2,
            Observation.FLORAL: 3,
        }

    def get_data_terminal_fate(self, year_no: int, code: Observation) -> List[float]:
        if year_no == 0:
            year_no = 1
        elif year_no < 0 or year_no > 6:
            year_no = 6
        if code in self.codes.keys():
            return self.data[(year_no, code)]
        else:
            raise ValueError(f"code must be in {self.codes.keys()}. {code} provided")

def _non_parametric_distribution(pdf):
    """Returns the index(x) at which the PDF(x) reaches random value in [0,1]"""
    import random
    r = random.random()
    s = 0
    for i, p in enumerate(pdf, 1):
        s += p
        if r <= s:
            return i
    return len(pdf)

def terminal_fate(
    year_no: int,
    observation: Observation,
    data_terminal_fate: Optional[TerminalFateData] = None,
) -> Observation:
    """This function returns a type of metamer (large, short, ...)
    :param year_no: is an int starting from 0 for the 1st year of the simulation
    :param observation: is a string. ['large', 'medium','small', 'floral'].
    """
    d = TerminalFate(data_terminal_fate)
    data = d.get_data_terminal_fate(year_no, observation)
    index = _non_parametric_distribution(data)
    if index == 1:
        return Observation.LARGE
    if index == 2:
        return Observation.MEDIUM
    if index == 3:
        return Observation.SMALL
    if index == 4:
        return Observation.FLORAL
    raise ValueError(f"Unknown index {index}")


def extract_sequences_to_df(input_file):
    """
    Extract all sequences from the input file and return them as a DataFrame.
    Each block in the file is parsed for sequence data, and only sequences of length >= 4 are kept.
    """
    with open(input_file, 'r') as f:
        content = f.read()
    pattern = r'((?:.|\n)*?)#\s*\(\d+\)'
    blocks = re.findall(pattern, content)
    print(f"Number of blocks found: {len(blocks)}")

    type_dict = {1: "LARGE", 2: "MEDIUM"}
    year_dict = {95: "Y1", 96: "Y2", 97: "Y3", 98: "Y4", 99: "Y5"}
    sequence_data = []
    for block in blocks:
        # Find all series of 7 numbers in the block
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
            'Terminal Fate': terminal_fate(int(year), type_dict.get(int(type_val), "UNKNOWN")),  # Add "Terminal Fate" column
        }
        sequence_data.append(sequence_data_line)

    df = pd.DataFrame(sequence_data)
    # Filter out sequences shorter than 4
    df_filtered = df[df['Sequence'].str.len() >= 4]

    # Print the number of filtered sequences
    filtered_count = len(df) - len(df_filtered)
    if filtered_count > 0:
        print(f"Removed {filtered_count} sequences that are too short (length < 4)")

    return df_filtered


def process_all_seq_files(directory='data'):
    """
    Process all .seq files in the specified directory and
    combine all extracted sequences into a single CSV file.
    """
    seq_files = [f for f in os.listdir(directory) if f.endswith('.seq')]

    if not seq_files:
        print("No .seq files found in the directory.")
        return

    total_sequences = 0
    processed_files = 0
    all_sequences_df = pd.DataFrame()

    # Process each .seq file
    for seq_file in seq_files:
        input_file = os.path.join(directory, seq_file)

        print(f"\nProcessing file: {seq_file}")
        try:
            df_sequences = extract_sequences_to_df(input_file)
            all_sequences_df = pd.concat([all_sequences_df, df_sequences])

            num_sequences = len(df_sequences)
            total_sequences += num_sequences
            processed_files += 1
            print(f"Extracted {num_sequences} sequences from {seq_file}")
        except Exception as e:
            print(f"Error processing {seq_file}: {str(e)}")

    # Save all sequences to a single CSV file
    output_file = "data/all_sequences.csv"
    all_sequences_df.to_csv(output_file, index=False)

    print(f"\nSummary: {total_sequences} sequences extracted from {processed_files} files.")
    print(f"All sequences have been saved to {output_file}")

    return total_sequences


def stats_from_csv(csv_file="data/all_sequences.csv"):
    """
    Load a CSV of sequences, compute descriptive statistics by (Observation, Year)
    pair, and return a summary DataFrame (displayed in full).
    """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    df = pd.read_csv(csv_file)

    # Compute the number of sequences and average length by (Observation, Year)
    stats = df.groupby(['Observation', 'Year']).agg(
        Number_of_sequences=('Sequence', 'count'),
        Average_length=('Sequence', lambda x: x.str.len().mean())
    ).reset_index()

    # Add global stats
    total_sequences = len(df)
    global_mean_length = df['Sequence'].str.len().mean()

    print(f"Total number of sequences: {total_sequences}")
    print(f"Global average sequence length: {global_mean_length:.2f}\n")

    print("Statistics by (Observation, Year):")
    print(stats)

    return stats


if __name__ == "__main__":
    # Display stats on the already preprocessed CSV
    stats_from_csv()


if __name__ == "__main__":
    # Process all .seq files in the current directory
    stats_from_csv()
