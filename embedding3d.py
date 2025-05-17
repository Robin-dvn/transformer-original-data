"""
This module provides a class to visualize transformer model embeddings in 3D space.
"""
import torch
import numpy as np
import pandas as pd
import itertools
import os
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import umap

class EmbeddingVisualizer:
    """
    A class to visualize embeddings from transformer models in 3D space.
    """

    def __init__(self, weights_path, embedding_dim=32):
        """
        Initialize the embedding visualizer.

        Args:
            weights_path (str): Path to the model weights file.
            embedding_dim (int): Dimension of the embedding vectors.
        """
        self.weights_path = weights_path
        self.embedding_dim = embedding_dim

        # Vocabulary setup
        self.vocab_to_id = {
            '<PAD>': 0, '< SOS >': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6,
            'DORMANT': 7, 'FLORAL': 8, 'LARGE': 9, 'MEDIUM': 10, 'SMALL': 11,
            'Y1': 12, 'Y2': 13, 'Y3': 14, 'Y4': 15, 'Y5': 16
        }
        self.id_to_vocab = {v: k for k, v in self.vocab_to_id.items()}
        self.vocab_size = len(self.vocab_to_id)
        self.pad_token_id = self.vocab_to_id['<PAD>']

        # Initialize embedding matrix
        self.embeddings_matrix = None

        # Define types and years for later use
        self.types = ['DORMANT', 'FLORAL', 'LARGE', 'MEDIUM', 'SMALL']
        self.years = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5']

        # Color configuration
        self.color_palette = px.colors.qualitative.Plotly
        self.type_color_map = {t: self.color_palette[i % len(self.color_palette)]
                              for i, t in enumerate(self.types)}

    def convert_tokens_to_ids(self, tokens):
        """
        Convert token strings to their corresponding ids.

        Args:
            tokens (list): List of token strings to convert.

        Returns:
            tuple: (ids, valid_tokens) - List of token ids and corresponding valid tokens.
        """
        ids = []
        valid_tokens = []
        for token in tokens:
            token_id = self.vocab_to_id.get(token, None)
            if token_id is not None:
                ids.append(token_id)
                valid_tokens.append(token)
            else:
                print(f"Warning: Token '{token}' not found in vocabulary. Skipped.")
        return ids, valid_tokens

    def load_embeddings(self):
        """
        Load embedding matrix from model weights file.

        Returns:
            bool: True if loading successful, False otherwise.
        """
        try:
            state_dict = torch.load(self.weights_path, map_location=torch.device('cpu'))
            embedding_key = 'embed.weight'  # Adjust if needed for your model

            # Extract from possible nested state dict
            if embedding_key not in state_dict:
                if 'model_state_dict' in state_dict:
                    print("Extracting from 'model_state_dict'.")
                    state_dict = state_dict['model_state_dict']
                elif 'model' in state_dict and isinstance(state_dict['model'], dict):
                    print("Extracting from 'model'.")
                    state_dict = state_dict['model']

            if embedding_key in state_dict:
                self.embeddings_matrix = state_dict[embedding_key].cpu().numpy()
                print(f"Embedding matrix successfully extracted via key '{embedding_key}'. Shape: {self.embeddings_matrix.shape}")
            else:
                print(f"Error: Embedding key '{embedding_key}' not found.")
                print("Available keys after potential unpacking:", state_dict.keys())
                return False

            # Verify embedding dimensions
            if self.embeddings_matrix.shape[0] != self.vocab_size:
                print(f"Warning: Embedding matrix size ({self.embeddings_matrix.shape[0]}) doesn't match vocab size ({self.vocab_size}).")
            if self.embeddings_matrix.shape[1] != self.embedding_dim:
                print(f"Warning: Extracted embedding dimension ({self.embeddings_matrix.shape[1]}) doesn't match expected dimension ({self.embedding_dim}).")

            return True

        except FileNotFoundError:
            print(f"Error: Weights file not found at {self.weights_path}")
            return False
        except Exception as e:
            print(f"An error occurred while loading or extracting weights: {e}")
            return False

    def prepare_visualization_data(self, use_sequences=False, tokens_to_visualize=None, sequences_to_visualize=None):
        """
        Prepare data for visualization by gathering relevant embeddings.

        Args:
            use_sequences (bool): If True, visualize sequences instead of individual tokens.
            tokens_to_visualize (list): List of tokens to visualize (used if use_sequences is False).
            sequences_to_visualize (list): List of token sequences to visualize (used if use_sequences is True).

        Returns:
            tuple: (vectors, labels, colors) - Data prepared for visualization.
        """
        if self.embeddings_matrix is None:
            print("Error: Embedding matrix not loaded. Call load_embeddings() first.")
            return None, None, None

        vectors_to_reduce = []
        labels = []
        point_colors = []

        # Default values if not provided
        if tokens_to_visualize is None:
            tokens_to_visualize = ['0', '1', '2', '3', '4']
        if sequences_to_visualize is None:
            sequences_to_visualize = list(itertools.product(self.types, self.years))

        if use_sequences:
            print("Preparing data for sequences...")
            temp_labels = [" ".join(seq) for seq in sequences_to_visualize]
            valid_indices = []
            sequence_types = []

            for i, seq in enumerate(sequences_to_visualize):
                token_ids, valid_tokens = self.convert_tokens_to_ids(list(seq))
                if len(token_ids) != len(seq):
                    print(f"Sequence '{' '.join(seq)}' ignored because some tokens are unknown.")
                    continue
                if any(tid >= self.embeddings_matrix.shape[0] for tid in token_ids):
                    print(f"Sequence '{' '.join(seq)}' ignored because some token IDs are out of bounds (max id: {self.embeddings_matrix.shape[0]-1}).")
                    continue

                seq_embeddings = self.embeddings_matrix[token_ids]
                mean_embedding = np.mean(seq_embeddings, axis=0)
                vectors_to_reduce.append(mean_embedding)
                valid_indices.append(i)
                sequence_types.append(seq[0])  # Store type (first element of sequence)

            # Keep only labels for valid sequences
            labels = [temp_labels[i] for i in valid_indices]
            vectors_to_reduce = np.array(vectors_to_reduce)
            # Generate color list based on types
            point_colors = [self.type_color_map.get(t, 'black') for t in sequence_types]

        else:  # Individual tokens
            print("Preparing data for individual tokens...")
            token_ids, valid_tokens = self.convert_tokens_to_ids(tokens_to_visualize)
            final_token_ids = []
            final_valid_tokens = []

            for tid, token in zip(token_ids, valid_tokens):
                if tid >= self.embeddings_matrix.shape[0]:
                    print(f"Token '{token}' (ID: {tid}) ignored because ID is out of bounds (max id: {self.embeddings_matrix.shape[0]-1}).")
                else:
                    final_token_ids.append(tid)
                    final_valid_tokens.append(token)

            if not final_token_ids:
                print("No valid tokens found for visualization.")
                return None, None, None

            vectors_to_reduce = self.embeddings_matrix[final_token_ids]
            labels = final_valid_tokens
            # Simple palette for individual tokens
            point_colors = self.color_palette[:len(labels)]

        # Verify we have enough data points
        if len(vectors_to_reduce) < 2:
            print("Not enough valid points (less than 2) to calculate similarity or dimensionality reduction.")
            return None, None, None
        else:
            print(f"{len(vectors_to_reduce)} valid vectors ready for analysis.")

        return np.array(vectors_to_reduce), labels, point_colors

    def calculate_similarity(self, vectors, labels):
        """
        Calculate and display cosine similarity between vectors.

        Args:
            vectors (numpy.ndarray): Matrix of vectors to compare.
            labels (list): Labels for each vector.

        Returns:
            pandas.DataFrame: Similarity matrix as a DataFrame.
        """
        if vectors is None or len(vectors) < 2:
            print("Not enough vectors to calculate similarity.")
            return None

        print("\nCalculating cosine similarity matrix...")
        similarity_matrix = cosine_similarity(vectors)
        similarity_df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)

        print("Cosine Similarity Table:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(similarity_df.round(3))

        return similarity_df

    def plot_similarity_heatmap(self, similarity_df, title="Cosine Similarity Heatmap between Embeddings"):
        """
        Plot a heatmap of similarity values.

        Args:
            similarity_df (pandas.DataFrame): Similarity matrix.
            title (str): Title for the heatmap.
        """
        if similarity_df is None:
            print("No similarity data to plot.")
            return

        print("\nGenerating cosine similarity heatmap...")

        fig_heatmap = px.imshow(
            similarity_df,
            text_auto=".2f",  # Display values on heatmap, rounded to 2 decimals
            aspect="auto",    # Adjust aspect to fill space
            color_continuous_scale='Viridis',  # Color scale
            title=title
        )

        fig_heatmap.update_xaxes(side="bottom")
        fig_heatmap.update_layout(
            width=800,
            height=700
        )

        fig_heatmap.show()

    def reduce_dimensions(self, vectors, method='umap', perplexity=30):
        """
        Reduce dimensions of vectors for visualization.

        Args:
            vectors (numpy.ndarray): Matrix of vectors to reduce.
            method (str): Method to use ('umap' or 'tsne').
            perplexity (int): Perplexity parameter for t-SNE (ignored for UMAP).

        Returns:
            numpy.ndarray: Reduced vectors in 3D space.
        """
        if vectors is None or len(vectors) < 2:
            print("Not enough vectors for dimensionality reduction.")
            return None

        if method.lower() == 'tsne':
            perplexity_value = min(perplexity, vectors.shape[0] - 1)
            if perplexity_value <= 0:
                print("Perplexity must be positive. Not enough points for t-SNE.")
                return None

            print(f"\nRunning t-SNE in 3D with {vectors.shape[0]} points and perplexity={perplexity_value}...")
            tsne_3d = TSNE(n_components=3, random_state=42,
                          perplexity=perplexity_value, n_iter=1000,
                          init='pca', learning_rate='auto')
            embeddings_3d = tsne_3d.fit_transform(vectors)
            print("3D t-SNE completed.")

        elif method.lower() == 'umap':
            print(f"\nRunning UMAP in 3D with {vectors.shape[0]} points and metric='cosine'...")
            reducer = umap.UMAP(n_components=3, metric='cosine', random_state=42)
            embeddings_3d = reducer.fit_transform(vectors)
            print("3D UMAP completed.")

        else:
            print(f"Unknown dimensionality reduction method: {method}")
            return None

        return embeddings_3d

    def plot_3d_embeddings(self, embeddings_3d, labels, colors, method='UMAP', use_sequences=False):
        """
        Create an interactive 3D plot of the embeddings.

        Args:
            embeddings_3d (numpy.ndarray): 3D vectors to plot.
            labels (list): Labels for each point.
            colors (list): Colors for each point.
            method (str): Method used for dimensionality reduction.
            use_sequences (bool): Whether sequences or individual tokens are being plotted.
        """
        if embeddings_3d is None:
            print("No embedding data to plot.")
            return

        print(f"Generating interactive 3D {method} plot with Plotly...")

        fig = go.Figure()

        # Add points and labels
        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=5,
                color=colors,
                opacity=0.8
            ),
            text=labels,
            textposition='top center',
            hoverinfo='text'
        ))

        # Format the 3D graph
        fig.update_layout(
            title=f"3D {method} Visualization of Embeddings ({'Sequences (colored by type)' if use_sequences else 'Tokens'})",
            scene=dict(
                xaxis_title=f"{method} Dimension 1",
                yaxis_title=f"{method} Dimension 2",
                zaxis_title=f"{method} Dimension 3"
            ),
            width=1000,
            height=800,
            margin=dict(r=20, b=10, l=10, t=50),
            showlegend=False
        )

        fig.show()

    def plot_embedding_matrix_heatmap(self):
        """
        Plot a heatmap of the full embedding matrix.
        """
        if self.embeddings_matrix is None:
            print("No embedding matrix to plot.")
            return

        print("\nGenerating numeric and colored heatmap of the complete embedding matrix...")

        # Use token names as labels for the Y axis
        y_labels = [self.id_to_vocab.get(i, f"ID_{i}") for i in range(self.embeddings_matrix.shape[0])]

        fig_embed_heatmap = px.imshow(
            self.embeddings_matrix,
            labels=dict(x="Embedding Dimension", y="Token", color="Value"),
            y=y_labels,
            text_auto=".2f",  # Display numeric values rounded to 2 decimals
            aspect="auto",
            color_continuous_scale='Viridis',
            title="Embedding Matrix Values Heatmap"
        )

        fig_embed_heatmap.update_layout(
            width=1200,
            height=700,
            xaxis_title="Embedding Dimension (0-31)",
            yaxis_title="Token"
        )

        fig_embed_heatmap.show()

    def analyze_digits_vs_sequences(self):
        """
        Compare digit embeddings with sequence embeddings.
        """
        if self.embeddings_matrix is None:
            print("No embedding matrix loaded. Call load_embeddings() first.")
            return

        print("\n--- Analyzing Digits vs Sequences (Type, Year) ---")

        # 1. Get embeddings for digits
        digit_tokens = ['0', '1', '2', '3', '4']
        digit_ids, valid_digit_tokens = self.convert_tokens_to_ids(digit_tokens)

        # Filter invalid or out-of-bounds IDs
        valid_digit_ids = []
        final_valid_digit_tokens = []
        for tid, token in zip(digit_ids, valid_digit_tokens):
            if tid >= self.embeddings_matrix.shape[0]:
                print(f"Digit '{token}' (ID: {tid}) ignored because ID is out of bounds (max id: {self.embeddings_matrix.shape[0]-1}).")
            else:
                valid_digit_ids.append(tid)
                final_valid_digit_tokens.append(token)

        if not valid_digit_ids:
            print("No valid digit embeddings found for comparison.")
            return

        # 2. Get embeddings for sequences
        sequences = list(itertools.product(self.types, self.years))
        seq_vectors, seq_labels, _ = self.prepare_visualization_data(
            use_sequences=True,
            sequences_to_visualize=sequences
        )

        if seq_vectors is None:
            print("Failed to prepare sequence data.")
            return

        # 3. Extract sequence types
        sequence_types = [seq.split()[0] for seq in seq_labels]

        # 4. Get digit embeddings
        digit_embeddings = self.embeddings_matrix[valid_digit_ids]

        # 5. Combine embeddings and labels
        combined_embeddings = np.vstack((digit_embeddings, seq_vectors))
        combined_labels = final_valid_digit_tokens + seq_labels
        combined_categories = ['Digit'] * len(final_valid_digit_tokens) + sequence_types

        # 6. Calculate combined similarity
        combined_similarity_df = self.calculate_similarity(combined_embeddings, combined_labels)

        # 7. Plot combined similarity heatmap
        self.plot_similarity_heatmap(
            combined_similarity_df,
            title="Cosine Similarity Heatmap: Digits vs Sequences (Type, Year)"
        )

        # 8. Reduce dimensions and plot 3D
        embeddings_combined_3d = self.reduce_dimensions(combined_embeddings, method='umap')

        if embeddings_combined_3d is not None:
            # 9. Define color mapping for combined categories
            unique_combined_categories = ['Digit'] + self.types
            combined_color_palette_list = ['#808080'] + [self.type_color_map.get(t, '#000000') for t in self.types]
            combined_category_color_map = {cat: combined_color_palette_list[i % len(combined_color_palette_list)]
                                           for i, cat in enumerate(unique_combined_categories)}
            point_colors_combined = [combined_category_color_map.get(cat, '#000000') for cat in combined_categories]

            # 10. Plot combined 3D visualization
            self.plot_3d_embeddings(
                embeddings_combined_3d,
                combined_labels,
                point_colors_combined,
                method='UMAP',
                use_sequences=True
            )


def main():
    """
    Main function to run the embedding visualization.
    """

    # Configuration
    weights_path = "experiments/DO_NBL-15_DM-32_DFF-1024_TS-20250422-091237_optuna/model_state.pth"  # Adjust to your file path
    embedding_dim = 32

    # Create visualizer
    visualizer = EmbeddingVisualizer(weights_path, embedding_dim)

    # Load embeddings
    if not visualizer.load_embeddings():
        print("Failed to load embeddings. Exiting.")
        return

    # Plot embedding matrix heatmap
    visualizer.plot_embedding_matrix_heatmap()

    # Visualize individual tokens
    tokens_to_visualize = ['0', '1', '2', '3', '4']
    vectors, labels, colors = visualizer.prepare_visualization_data(
        use_sequences=False,
        tokens_to_visualize=tokens_to_visualize
    )

    if vectors is not None:
        # Calculate and plot token similarity
        similarity_df = visualizer.calculate_similarity(vectors, labels)
        visualizer.plot_similarity_heatmap(similarity_df)

        # Reduce dimensions and plot
        embeddings_3d = visualizer.reduce_dimensions(vectors, method='umap')
        if embeddings_3d is not None:
            visualizer.plot_3d_embeddings(embeddings_3d, labels, colors, method='UMAP', use_sequences=False)

    # Visualize sequences
    vectors, labels, colors = visualizer.prepare_visualization_data(
        use_sequences=True
    )

    if vectors is not None:
        # Calculate and plot sequence similarity
        similarity_df = visualizer.calculate_similarity(vectors, labels)
        visualizer.plot_similarity_heatmap(similarity_df, title="Cosine Similarity Heatmap between Sequence Embeddings")

        # Reduce dimensions and plot
        embeddings_3d = visualizer.reduce_dimensions(vectors, method='umap')
        if embeddings_3d is not None:
            visualizer.plot_3d_embeddings(embeddings_3d, labels, colors, method='UMAP', use_sequences=True)

    # Compare digits and sequences
    visualizer.analyze_digits_vs_sequences()

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
