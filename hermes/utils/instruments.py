"""Instrument Classes."""
from enum import Enum
from pathlib import Path
from typing import Union
import numpy as np

from .ml import GPCClassifier, HSGPCClassifier, Classifier
from .cluster import cluster
import orix
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score

import re

r = re.compile("\d+$")


class EBSDModelsEnum(str, Enum):
    Heteroscedastic = "Heteroscedastic"
    Homoscedastic = "Homoscedastic"


# TODO: add simulation mode to all


class DataWrangler:
    """Base class for data wranglers"""

    # bluesky??

    def choose_window(self, x_seed, y_seed, input_locs, window_size=51):
        """
        input_locs is the locations of all the data
        x in column 0, y in column 1, z in column 2
        We want data that is a square window of size window_size^2 entries
        """
        # only include the locations where the x location is in renge definde by the index from x_seed to x_seed+window_size
        x_input_locs = input_locs[
            np.isin(
                input_locs[:, 0],
                np.unique(input_locs[:, 0])[x_seed : window_size + x_seed],
            )
        ]
        mask = np.isin(
            input_locs[:, 0],
            np.unique(input_locs[:, 0])[x_seed : window_size + x_seed],
        )
        # Down select for the values of y defined by the similar range.
        y_input_locs = x_input_locs[
            np.isin(
                x_input_locs[:, 1],
                np.unique(x_input_locs[:, 1])[y_seed : window_size + y_seed],
            )
        ]
        mask = mask * np.isin(
            input_locs[:, 1],
            np.unique(input_locs[:, 1])[y_seed : window_size + y_seed],
        )
        return y_input_locs, mask

    def euler_to_quat_noise(self, euler_angles, sigma_psi1, sigma_phi, sigma_psi2):
        # Given Euler angles in Bunge Convention (psi1, phi, psi2),
        # and some Gausian noise on those angles,
        # Calculate the noise in the quaternions:

        noise_psi1 = np.random.normal(0, sigma_psi1)
        noise_phi = np.random.normal(0, sigma_phi)
        noise_psi2 = np.random.normal(0, sigma_psi2)

        psi1 = euler_angles[:, 0].reshape(-1, 1) + noise_psi1
        phi = euler_angles[:, 1].reshape(-1, 1) + noise_phi
        psi2 = euler_angles[:, 2].reshape(-1, 1) + noise_psi2

        q0 = np.cos(0.5 * phi) * np.cos(0.5 * (psi1 + psi2))

        q1 = -np.sin(0.5 * phi) * np.cos(0.5 * (psi1 - psi2))

        q2 = -np.sin(0.5 * phi) * np.sin(0.5 * (psi1 - psi2))

        q3 = -np.cos(0.5 * phi) * np.sin(0.5 * (psi1 + psi2))

        p = q0 / np.abs(q0)  # if q0 is negative, flip the sign of the quaternion

        noisy_quaternions = p * np.concatenate((q0, q1, q2, q3), 1)

        return noisy_quaternions

    # def cluster(self, inputs, measurements, scale, resolution):
    #     cluster(inputs, measurements, scale, resolution)


class XRD(DataWrangler):
    """Instrument type class"""

    pass


class XRDEmulator(XRD):
    pass


class SEM(DataWrangler):
    """Instrument type class"""

    pass


class EBSD(DataWrangler):
    """Instrument type class"""

    def load_ground_truth(self, path: Union[Path, str], names: list):
        self.ground_truth = pd.read_fwf(path, names=names)

    def load_ang(
        self, path: Union[Path, str], initial_layer_depth=-0.4, depth_step=0.2
    ):
        # Assuming the last few characters are the layers
        layer = int(r.search(str.split(".")[0])[0]) - 1

        # Read .ang file
        if isinstance(path, str):
            path = Path(path)
        if not isinstance(path, Path):
            raise TypeError("path must be a Path or a string representing a path")

        assert path.suffix == ".ang", "Invalid ang file"
        self.cm = orix.io.loadang(path)

        # Read each column from the .ang file
        (
            self.euler1,
            self.euler2,
            self.euler3,
            self.x,
            self.y,
            self.iq,
            self.ci,
            self.phase_id,
            self.di,
            self.fit,
        ) = np.loadtxt(path, unpack=True)
        self.measurements = np.concatenate(
            (
                self.euler1.reshape(-1, 1),
                self.euler2.reshape(-1, 1),
                self.euler3.reshape(-1, 1),
            ),
            1,
        )

        # List the xyz locations
        z = layer * depth_step + initial_layer_depth
        input_loc = np.concatenate(
            (self.x.reshape(-1, 1), self.y.reshape(-1, 1)), axis=1
        )
        self.real_locations = np.concatenate(
            (input_loc, z * np.ones_like(self.x.reshape(-1, 1))), axis=1
        )

        return self.real_locations, self.measurements

    @property
    def ground_truth_labels(self):
        layer_ground_truth = self.gound_gruth[self.ground_truth["Z"] == -0.4]
        # layer_ground_truth = layer_ground_truth[layer_ground_truth['X'] >= 0-10e-3]
        # layer_ground_truth = layer_ground_truth[layer_ground_truth['Y'] >= 0-10e-3]
        # layer_ground_truth = layer_ground_truth[layer_ground_truth['X'] <= np.max(input_loc[:,0])+10e-3]
        # layer_ground_truth = layer_ground_truth[layer_ground_truth['Y'] <= np.max(input_loc[:,1])+10e-3]
        layer_ground_truth = layer_ground_truth[
            np.isin(layer_ground_truth["X"], self.input_loc[:, 0])
        ]
        layer_ground_truth = layer_ground_truth[
            np.isin(layer_ground_truth["Y"], self.input_loc[:, 1])
        ]
        ground_truth_labels = np.array(layer_ground_truth["Grain Label"])
        return ground_truth_labels

    def run_al_campaign(
        self, model_type: EBSDModelsEnum, params, maximum_loops: int = 30
    ):
        random_seed = params[1]
        queue = params[2]
        np.random.seed(random_seed)

        window = 75  # pixels in the square window to use
        x_lim = np.unique(self.input_loc[:, 0]).shape[0]
        y_lim = np.unique(self.input_loc[:, 1]).shape[0]
        assert x_lim - window > 0, "Window too large for x input size"
        assert y_lim - window > 0, "Window too large for y input size"

        x_seed = np.random.randint(0, x_lim - window + 1)
        y_seed = np.random.randint(0, y_lim - window + 1)

        sub_input_loc, mask = self.choose_window(
            x_seed, y_seed, self.input_loc, window_size=window
        )

        local_ground_truth = self.ground_truth_labels[mask]

        # Take noisy observations!
        # with ## uncertainty in the Euler angles
        local_euler = self.data[mask]
        local_data = self.Euler_to_quat_noise(
            local_euler, 2 * np.pi / 180, 2 * np.pi / 180, 2 * np.pi / 180
        )
        local_data = orix.quaternion.Quaternion(data=local_data)
        # local_data = data[mask]

        # Number of points measured to start with
        start_measurements = 10
        measured_index = np.random.permutation(sub_input_loc.shape[0])
        measured_index = measured_index[np.arange(0, start_measurements)]

        # Start a container for the results tabel
        al_results_table_values = []

        for i in range(maximum_loops):
            # Get the Active training locations from the index:
            active_train_locations = sub_input_loc[measured_index]

            # Take measurements at the active training sites:
            active_train_measurements = local_data[measured_index]

            # Cluster
            (
                active_labels,
                active_probabilities,
                active_c,
                active_graph,
            ) = cluster(active_train_locations, active_train_measurements)

            __models = {
                "Homoscedastic": GPCClassifier,
                "Heteroscedastic": HSGPCClassifier,
            }

            # Train
            def __prob(cl, prob):
                if cl == "Homoscedastic":
                    return len(prob[0, :])
                return prob

            # TODO might need: return object (not attr model)
            active_model = __models[model_type](
                active_train_locations,
                active_labels,
                __prob(model_type, active_probabilities),
            ).model
            # Predict
            # TODO currently working
            (
                active_classes,
                active_total_var,
                active_mean,
                active_var,
            ) = Classifier.predict_class(active_model, sub_input_loc)
            # Caclulare Adjusted Rand Score to the ground truth
            ars = adjusted_rand_score(local_ground_truth, active_classes)

            # Acquire next points
            points = Classifier.acquire_next_batch(
                active_model, sub_input_loc, measured_index, batch_size=4
            )

            loop_results = [
                i,
                measured_index,
                points,
                active_classes,
                active_total_var,
                ars,
            ]
            al_results_table_values.append(loop_results)

            measured_index = np.concatenate((measured_index, points.flatten()))

            # Test for convergence
            if i > 3:
                # find the active_classes of previous result
                back_1_map = al_results_table_values[-2][3]
                back_2_map = al_results_table_values[-3][3]
                back_3_map = al_results_table_values[-4][3]

                ars_back = np.array(
                    [
                        adjusted_rand_score(active_classes, back_1_map),
                        adjusted_rand_score(active_classes, back_2_map),
                        adjusted_rand_score(active_classes, back_3_map),
                    ]
                )

                # If all the ARS scores to the previous 4 loops are above a value, escape!
                converged = ars_back > 0.85
                if all(converged):
                    break

        model_tabel_values = [
            model_type,
            random_seed,
            local_ground_truth,
            x_seed,
            y_seed,
            al_results_table_values,
        ]

        queue.put(model_tabel_values)
        return model_tabel_values


class BrukerD8(XRD):
    """Bruker D8 instrument-specific class"""

    pass


class Ricago234(XRD):
    """Ricago 234 instrument-specific class"""

    pass
