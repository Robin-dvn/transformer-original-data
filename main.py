"""
This module is the main entry point for running training and validation pipelines.
It allows for testing with a basic configuration or running the full version with multiple configurations.
"""

from pipeline import train_generate_validate_pipeline

if __name__ == "__main__":
    # Variable to control whether to run a test or a full version
    test = True  # Set to False for the full version

    if test:
        # Basic configuration for testing
        configs = [(15, 32, 1024)]  # Only the first configuration for testing
        nb_epochs = 1  # Reduced to 1 epoch for testing
        sync = False
        print("=== Test Mode ===")
    else:
        # Full basic configuration
        configs = [
            # Basic configuration with d_model = 32
            (15, 32, 1024),  # NBL-15_DM-32_DFF-1024
        ]
        nb_epochs = 2800
        sync = True
        print("=== Full Mode ===")

    for nb_layers, d_model, dim_feedforward in configs:
        config_dict = {
            'dataset_path': "data/all_sequences.csv",
            'seed': 42,
            'batch_size': 128,
            'val_split': 0.8,
            'vocab_size': 17,
            'padding_idx': 0,
            'n_head': 4,
            'd_model': d_model,
            'nb_layers': nb_layers,
            'lr': 5e-6,
            'nb_epoch': nb_epochs,
            'dim_feedforward': dim_feedforward,
            'dynamic': False,
            'scheduler': {
                'name': 'None',
                'params': {}
            },
            'early_stopping': {
                'name': 'None',
                'params': {}
            },
            'continue_training': False,
            'checkpoint_path' :'experiments/DO_NBL-15_DM-32_DFF-1024_TS-20250422-134740/model_state.pth',
            'auto_precision': False,
            'graph_name': f"DO_NBL-{nb_layers}_DM-{d_model}_DFF-{dim_feedforward}_baseline"
        }
        # Baseline training example
        print(f"=== {'Test' if test else 'Trial'} baseline ===")
        validator_baseline = train_generate_validate_pipeline(config_dict,sync_wandb=sync)

        ################## Training examples #####################

        # # Example: cyclical learning rate
        # config_cyclical = config_dict.copy()
        # config_cyclical["scheduler"] = {
        #     "name": "cyclical",
        #     "params": {"base_lr": 5e-8, "max_lr": 5e-4, "step_size_up": 782}
        # }
        # config_cyclical["graph_name"] = f"DO_NBL-{nb_layers}_DM-{d_model}_DFF-{dim_feedforward}_cyclical"

        # print(f"=== {'Test' if test else 'Trial'} cyclical learning rate ===")
        # validator_cyclical = train_generate_validate_pipeline(config_cyclical,sync_wandb=sync)

        # # Example: early stopping
        # config_early = config_dict.copy()
        # config_early["scheduler"] = {
        #     "name": "None",
        #     "params": {}
        # }
        # config_early["early_stopping"] = {
        #     "name": "patience",
        #     "params": {"patience": 100, "verbose": True, "delta": 0.005}
        # }
        # config_early["graph_name"] = f"DO_NBL-{nb_layers}_DM-{d_model}_DFF-{dim_feedforward}_early"
        # print(f"=== {'Test' if test else 'Trial'} early stopping ===")
        # validator_early = train_generate_validate_pipeline(config_early,sync_wandb=sync)
