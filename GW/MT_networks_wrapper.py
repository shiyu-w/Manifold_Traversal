import numpy as np
import pickle
import time
from models.manifold_traversal import ManifoldTraversal



class MTNetworksWrapper:
    """
    Conducts ablation study on GW data using manifold traversal networks.
    """

    def __init__(self, X_train, X_natural_train, X_test, X_natural_test, sigma, d):
        """
        Initialize ablation study.

        Args:
            data_dir: Directory containing GW training/test data
            save_dir: Directory to save results
        """

        self.sigma = sigma
        self.d = d

        
        self.X_train = X_train
        self.X_natural_train = X_natural_train
        self.X_test = X_test
        self.X_natural_test = X_natural_test
        self.N_train = self.X_train.shape[1]
        self.N_test = self.X_test.shape[1]
        self.D = self.X_train.shape[0]

        self.networks = []
        self.network_names = []
        self.network_results = []
        self.network_stats = []
        self.analysis_results = []


    def get_hyperparameter_configs(self):
        """
        Get the same 12 hyperparameter configurations as the original ablation study.

        Returns:
            List of dictionaries containing hyperparameters for each network
        """
        configs = []

        sigma_sq_D = self.sigma ** 2 * self.D
        sigma_sq_d = self.sigma ** 2 * self.d

        base_configs = [
            # Network 1
            {
                'R_is_const': False,
                'R_denoising': np.sqrt(2.06 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 2
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(2.06 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 3
            {
                'R_is_const': False,
                'R_denoising': np.sqrt(2.06 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(8 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 4
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(2.75 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.75 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 5
            {
                'R_is_const': False,
                'R_denoising': np.sqrt(2.06 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.3,
                'exp_coeff': 1 / 3
            },
            # Network 6
            {
                'R_is_const': False,
                'R_denoising': np.sqrt(2.06 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(4 * sigma_sq_d),
                'prod_coeff': 1.15,
                'exp_coeff': 1 / 2
            },
            # Network 7
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(2.39 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.75 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 8
            {
                'R_is_const': False,
                'R_denoising': np.sqrt(2.06 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(30 * sigma_sq_d),
                'prod_coeff': 1.5,
                'exp_coeff': 1 / 2
            },
            # Network 9
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(2 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 10
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(2.19 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 11
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(3.13 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(3.53 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 12
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(1.94 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            }
        ]

        return base_configs

    def train_networks(self, batch_size=4000, save_path=None):
        """
        Train all networks with different hyperparameter configurations.

        Args:
            batch_size: Batch size for training progress reporting
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        configs = self.get_hyperparameter_configs()

        print(f"Training {len(configs)} networks with different hyperparameters...")
        print("=" * 80)

        for i, config in enumerate(configs):
            network_name = f"NETWORK_{i + 1}"
            print(f"\nTraining {network_name}")
            print(f"Config: {config}")
            print("-" * 60)

            mt = ManifoldTraversal(
                intrinsic_dim=self.d,
                ambient_dim=self.D,
                sigma=self.sigma,
                **config
            )

            start_time = time.time()
            results = mt.fit(self.X_train, self.X_natural_train,
                             batch_size=batch_size, verbose=True)
            training_time = time.time() - start_time

            self.networks.append(mt)
            self.network_names.append(network_name)
            self.network_results.append(results)

            stats = mt.network.get_network_stats()
            stats['training_time'] = training_time
            stats['config'] = config
            self.network_stats.append(stats)

            print(f"Training completed in {training_time:.2f}s")
            print(f"Network stats: {stats}")
            print("~" * 60)

        print(f"\nAll {len(configs)} networks trained successfully!")


        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(self, f)

                

    def analyze_networks(self, num_test_samples=None):
        """
        Analyze performance of all trained networks.

        Args:
            num_test_samples: Number of test samples to use (None for all)
        """
        if not self.networks:
            raise ValueError("No networks trained. Call train_networks() first.")

        if num_test_samples is None:
            num_test_samples = self.N_test

        print(f"\nAnalyzing performance on {num_test_samples} test samples...")
        print("=" * 80)

        analysis_results = []

        for i, (mt, name) in enumerate(zip(self.networks, self.network_names)):
            print(f"Analyzing {name}...")

            results = mt.analyze_performance(
                self.X_test, self.X_natural_test,
                num_samples=num_test_samples
            )

            analysis_results.append(results)

            print(f"  Exhaustive Search: Error={results['exhaustive']['avg_distance']:.6f}, "
                  f"Complexity={results['exhaustive']['avg_mults']:.1f}")
            print(f"  Mixed Order (MT): Error={results['mixed_order']['avg_distance']:.6f}, "
                  f"Complexity={results['mixed_order']['avg_mults']:.1f}")
            print(f"  First Order Only: Error={results['first_order_only']['avg_distance']:.6f}, "
                  f"Complexity={results['first_order_only']['avg_mults']:.1f}")
            print(f"  Zero Order Only: Error={results['zero_order_only']['avg_distance']:.6f}, "
                  f"Complexity={results['zero_order_only']['avg_mults']:.1f}")
            print()

        self.analysis_results = analysis_results
        print("Analysis complete!")




