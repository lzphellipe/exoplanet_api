import pandas as pd
import lightkurve as lk
import numpy as np
import os
import logging
import json

logger = logging.getLogger(__name__)


class LightkurveService:
    def __init__(self, data_dir: str = "data"):
        """Initialize LightkurveService with directories and logging.

        Args:
            data_dir (str): Directory for data storage. Defaults to 'data'.
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.lightcurve_dir = os.path.join(data_dir, "lightcurves")
        self.features_file = os.path.join(data_dir, "lightkurve_features.csv")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.lightcurve_dir, exist_ok=True)

    def preprocess_train_data(self, dataset_path: str, max_samples: int = None,
                              mission: str = "Kepler") -> pd.DataFrame:
        """Load and preprocess Kepler/TESS light curve data for training.

        Args:
            dataset_path (str): Path to the CSV/JSON file containing training data.
            max_samples (int, optional): Maximum number of samples to process.
            mission (str): Data mission (e.g., 'Kepler', 'TESS'). Defaults to 'Kepler'.

        Returns:
            pd.DataFrame: Processed light curve data with flux, label, and identifier.
        """
        self.logger.info(f"Preprocessing training dataset: {dataset_path}")
        try:
            # Load data
            if dataset_path.endswith(".json"):
                data_train = pd.read_json(dataset_path)
            else:
                data_train = pd.read_csv(dataset_path)

            # Define identifier and column names based on mission
            id_col = "kepid" if mission == "Kepler" else "hostname"
            period_col = "koi_period" if mission == "Kepler" else "pl_orbper"
            t0_col = "koi_time0bk" if mission == "Kepler" else "pl_tranmid"
            duration_col = "koi_duration" if mission == "Kepler" else "pl_trandur"
            label_col = "koi_disposition" if mission == "Kepler" else "disposition"

            # Keep relevant columns
            columns = [id_col, label_col, period_col, t0_col, duration_col]
            if mission == "Kepler":
                columns.append("koi_quarters")
            data_train = data_train[columns].copy()

            # Filter for confirmed planets and false positives, drop NaNs
            data_train = data_train[data_train[label_col].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
            data_train = data_train.dropna().reset_index(drop=True)
            if max_samples:
                data_train = data_train.head(max_samples)

            # Extract columns
            identifiers = data_train[id_col].values
            dispositions = data_train[label_col].values
            periods = data_train[period_col].values
            t0s = data_train[t0_col].values
            durations = data_train[duration_col].values

            curves_finals = []
            identifiers_finals = []
            labels_finals = []

            # Process each target
            for idx, (identifier, disposition, period, t0, duration) in enumerate(
                    zip(identifiers, dispositions, periods, t0s, durations)):
                try:
                    # Download light curves
                    lcs = lk.search_lightcurve(str(identifier), mission=mission, cadence='long').download_all()
                    if lcs is None:
                        self.logger.warning(f"No light curves for {identifier} (index: {idx})")
                        continue

                    # Ensure lcs is a list
                    if isinstance(lcs, lk.LightCurveCollection):
                        lcs = list(lcs)
                    elif isinstance(lcs, lk.LightCurve):
                        lcs = [lcs]

                    # Preprocess light curves
                    lcs = [lc.remove_nans().remove_outliers(sigma=3).normalize() for lc in lcs]
                    flattened_curves = []

                    for lc in lcs:
                        temp_fold = lc.fold(period, epoch_time=t0)
                        fractional_duration = (duration / 24.0) / period
                        phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
                        transit_mask = np.isin(lc.time.value, temp_fold.time_original.value[phase_mask])
                        lc_flat, _ = lc.flatten(return_trend=True, mask=transit_mask)
                        flattened_curves.append(lc_flat)

                    # Stitch, fold, and bin
                    lc_stitched = lk.LightCurveCollection(flattened_curves).stitch()
                    lc_fold = lc_stitched.fold(period, epoch_time=t0)
                    lc_final = lc_fold.bin(bins=50)

                    # Standardize flux
                    flux_mean = np.nanmean(lc_final.flux)
                    flux_std = np.nanstd(lc_final.flux)
                    if flux_std == 0 or np.isnan(flux_std):
                        curve_final = np.array(lc_final.flux.value - flux_mean, dtype=float)
                    else:
                        curve_final = np.array((lc_final.flux.value - flux_mean) / flux_std, dtype=float)

                    curves_finals.append(curve_final)
                    identifiers_finals.append(identifier)
                    labels_finals.append(disposition)

                except Exception as e:
                    self.logger.error(f"Error processing {identifier} (index: {idx}): {e}")

            # Create DataFrame
            data_train_final = pd.DataFrame(curves_finals, dtype=float)
            data_train_final = data_train_final.interpolate(axis=1)
            data_train_final["identifier"] = identifiers_finals
            data_train_final["label"] = labels_finals

            data_train_final.to_csv(self.features_file, index=False)
            self.logger.info(f"Training features saved to {self.features_file} with {len(data_train_final)} records")
            return data_train_final
        except Exception as e:
            self.logger.error(f"Error preprocessing training dataset {dataset_path}: {e}")
            raise

    def preprocess_test_data(self, dataset_path: str, max_samples: int = None, mission: str = "Kepler") -> pd.DataFrame:
        """Load and preprocess Kepler/TESS light curve data for testing.

        Args:
            dataset_path (str): Path to the CSV/JSON file containing test data.
            max_samples (int, optional): Maximum number of samples to process.
            mission (str): Data mission (e.g., 'Kepler', 'TESS'). Defaults to 'Kepler'.

        Returns:
            pd.DataFrame: Processed light curve data with flux and identifier.
        """
        self.logger.info(f"Preprocessing test dataset: {dataset_path}")
        try:
            # Load data
            if dataset_path.endswith(".json"):
                data_test = pd.read_json(dataset_path)
            else:
                data_test = pd.read_csv(dataset_path)

            # Define column names
            id_col = "kepid" if mission == "Kepler" else "hostname"
            period_col = "koi_period" if mission == "Kepler" else "pl_orbper"
            t0_col = "koi_time0bk" if mission == "Kepler" else "pl_tranmid"
            duration_col = "koi_duration" if mission == "Kepler" else "pl_trandur"

            # Keep relevant columns
            columns = [id_col, period_col, t0_col, duration_col]
            if mission == "Kepler":
                columns.append("koi_quarters")
            data_test = data_test[columns].copy()

            # Drop NaNs
            data_test = data_test.dropna().reset_index(drop=True)
            if max_samples:
                data_test = data_test.head(max_samples)

            # Extract columns
            identifiers = data_test[id_col].values
            periods = data_test[period_col].values
            t0s = data_test[t0_col].values
            durations = data_test[duration_col].values

            curves_finals = []
            identifiers_finals = []

            # Process each target
            for idx, (identifier, period, t0, duration) in enumerate(zip(identifiers, periods, t0s, durations)):
                try:
                    # Download light curves
                    lcs = lk.search_lightcurve(str(identifier), mission=mission, cadence='long').download_all()
                    if lcs is None:
                        self.logger.warning(f"No light curves for {identifier} (index: {idx})")
                        continue

                    # Ensure lcs is a list
                    if isinstance(lcs, lk.LightCurveCollection):
                        lcs = list(lcs)
                    elif isinstance(lcs, lk.LightCurve):
                        lcs = [lcs]

                    # Preprocess light curves
                    lcs = [lc.remove_nans().remove_outliers(sigma=3).normalize() for lc in lcs]
                    flattened_curves = []

                    for lc in lcs:
                        temp_fold = lc.fold(period, epoch_time=t0)
                        fractional_duration = (duration / 24.0) / period
                        phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
                        transit_mask = np.isin(lc.time.value, temp_fold.time_original.value[phase_mask])
                        lc_flat, _ = lc.flatten(return_trend=True, mask=transit_mask)
                        flattened_curves.append(lc_flat)

                    # Stitch, fold, and bin
                    lc_stitched = lk.LightCurveCollection(flattened_curves).stitch()
                    lc_fold = lc_stitched.fold(period, epoch_time=t0)
                    lc_final = lc_fold.bin(bins=50)

                    # Standardize flux
                    flux_mean = np.nanmean(lc_final.flux)
                    flux_std = np.nanstd(lc_final.flux)
                    if flux_std == 0 or np.isnan(flux_std):
                        curve_final = np.array(lc_final.flux.value - flux_mean, dtype=float)
                    else:
                        curve_final = np.array((lc_final.flux.value - flux_mean) / flux_std, dtype=float)

                    curves_finals.append(curve_final)
                    identifiers_finals.append(identifier)

                except Exception as e:
                    self.logger.error(f"Error processing {identifier} (index: {idx}): {e}")

            # Create DataFrame
            data_test_final = pd.DataFrame(curves_finals, dtype=float)
            data_test_final = data_test_final.interpolate(axis=1)
            data_test_final["identifier"] = identifiers_finals

            data_test_final.to_csv(self.features_file, index=False)
            self.logger.info(f"Test features saved to {self.features_file} with {len(data_test_final)} records")
            return data_test_final
        except Exception as e:
            self.logger.error(f"Error preprocessing test dataset {dataset_path}: {e}")
            raise

    def periodogram_lightcurve(self, results_tests: pd.DataFrame, mission: str = "Kepler") -> list:
        """Generate BLS periodograms and folded light curves for candidate exoplanets.

        Args:
            results_tests (pd.DataFrame): DataFrame with 'identifier' and 'pred_label' columns.
                                         Rows where 'pred_label' == 1 are treated as exoplanet candidates.
            mission (str): Data mission (e.g., 'TESS', 'Kepler'). Defaults to 'Kepler'.

        Returns:
            list: List of dicts containing transit parameters and folded light curves for each target.
        """
        self.logger.info("Generating BLS periodograms for exoplanet candidates")
        try:
            # Filter only the rows classified as exoplanet candidates
            exoplanets = results_tests[results_tests['pred_label'] == 1].copy()
            list_exoplanets = []

            # Loop through each identifier (kepid or hostname)
            for identifier in exoplanets['identifier'].values:
                try:
                    # Download all available light curves for this target
                    lc_collection = lk.search_lightcurve(str(identifier), mission=mission).download_all()
                    if lc_collection is None or len(lc_collection) == 0:
                        self.logger.warning(f"No light curves found for {identifier}")
                        continue

                    # Stitch all available sectors into a single continuous light curve
                    lc = lc_collection.stitch()

                    # Compute BLS periodogram
                    periodogram = lc.to_periodogram(method='bls', frequency_factor=30)
                    periods_array = periodogram.period.value
                    power_array = periodogram.power.value

                    # Extract transit parameters
                    planet_period = periodogram.period_at_max_power
                    planet_t0 = periodogram.transit_time_at_max_power
                    planet_dur = periodogram.duration_at_max_power

                    # Fold the light curve
                    folded_lc = lc.fold(period=planet_period, epoch_time=planet_t0)
                    phase = folded_lc.phase.value
                    flux = folded_lc.flux.value

                    # Build result dictionary
                    lc_dict = {
                        'identifier': identifier,
                        'planet_period': float(planet_period.value),
                        'transit_t0': float(planet_t0.value),
                        'transit_duration': float(planet_dur.value),
                        'lightcurve': {
                            'phase': phase.tolist(),
                            'flux': flux.tolist()
                        },
                        'periodogram': {
                            'periods': periods_array.tolist(),
                            'power': power_array.tolist()
                        }
                    }
                    list_exoplanets.append(lc_dict)

                except Exception as e:
                    self.logger.error(f"Error processing {identifier}: {e}")

            self.logger.info(f"Processed {len(list_exoplanets)} exoplanet candidates")
            return list_exoplanets
        except Exception as e:
            self.logger.error(f"Error generating periodograms: {e}")
            raise