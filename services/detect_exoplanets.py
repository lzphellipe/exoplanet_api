import numpy as np
import pandas as pd
import json
import lightkurve as lk
from lightkurve import search_lightcurve, LightCurveCollection, LightCurve
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer

def preprocess_train_data(path_train):
  """
    Load and preprocess Kepler light curve data for training.

    This function:
    - Loads a CSV file with Kepler Object of Interest (KOI) data.
    - Filters for confirmed planets and false positives.
    - Cleans NaN values and resets the index.
    - Downloads and preprocesses the light curves:
        - Removes NaNs and outliers
        - Normalizes flux
        - Folds light curves based on period and transit time
        - Masks transits and flattens the light curve
        - Stitches multiple curves and bins them
        - Standardizes the flux values
    - Returns a DataFrame containing processed curves with corresponding labels and kepid.

    Parameters:
    ----------
    path_train : str
        Path to the CSV file containing KOI data.

    Returns:
    -------
    pd.DataFrame
        Processed light curve data with columns for flux, label, and kepid.
    """

  # Load training data from CSV
  data_train = pd.read_csv(path_train, sep = ",")

  # Keep only relevant columns
  data_train = data_train[['kepid','koi_disposition','koi_period','koi_time0bk','koi_duration','koi_quarters']]

  # Filter for confirmed planets and false positives, drop rows with missing values
  data_train = data_train[data_train.koi_disposition.isin(["CONFIRMED", "FALSE POSITIVE"])]
  data_train = data_train.dropna()
  data_train = data_train.reset_index(drop=True)

  # Extract individual columns as arrays
  kepids = data_train['kepid'].values
  dispositions = data_train['koi_disposition'].values
  periods = data_train['koi_period'].values
  t0s = data_train['koi_time0bk'].values
  durations = data_train['koi_duration'].values

  # Initialize lists to store processed data
  kepids_finals = []
  curves_finals = []
  labels_finals = []

  # Loop through each KOI to process its light curve
  for idx, (kepid, disposition, period, t0, duration) in enumerate(zip(kepids, dispositions, periods, t0s, durations)):

      try:
          # Download light curve(s) from Kepler
          lcs = search_lightcurve(str(kepid), author='Kepler', cadence='long').download_all()

          if lcs is None:
            print(idx, "not downloaded")
            continue

          # Ensure lcs is a list of LightCurve objects
          if isinstance(lcs, LightCurveCollection):
              lcs = list(lcs)
          elif isinstance(lcs, LightCurve):
              lcs = [lcs]

          # Remove NaNs and outliers
          lcs = [lc.remove_nans() for lc in lcs]
          lcs = [lc.remove_outliers(sigma=3) for lc in lcs]

          # Normalize flux
          normalize_lcs = [lc.normalize() for lc in lcs]

          flattened_curves = []

          for lc in normalize_lcs:
            # Fold the light curve using the period and transit epoch
            temp_fold = lc.fold(period, epoch_time=t0)
            fractional_duration = (duration / 24.0) / period
            phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
            transit_mask = np.isin(lc.time.value, temp_fold.time_original.value[phase_mask])

            # Flatten the light curve while masking the transit
            lc_flat, trend_lc = lc.flatten(return_trend=True, mask=transit_mask)
            flattened_curves.append(lc_flat)

          # Stitch all flattened light curves into one
          lc_stitched = LightCurveCollection(flattened_curves).stitch()

          # Fold and bin the stitched curve
          lc_fold = lc_stitched.fold(period, epoch_time=t0)
          lc_final = lc_fold.bin(bins=50)

          # Standardize flux
          flux_mean = np.nanmean(lc_final.flux)
          flux_std = np.nanstd(lc_final.flux)

          if flux_std == 0 or np.isnan(flux_std):
              curve_final = np.array(lc_final.flux.value - flux_mean, dtype=float)
          else:
              curve_final = np.array((lc_final.flux.value - flux_mean) / flux_std, dtype=float)

          # Append processed curve, label, and kepid
          curves_finals.append(curve_final)
          labels_finals.append(disposition)
          kepids_finals.append(kepid)

      except Exception as e:
          print(idx, kepid, e)

  # Convert processed curves to DataFrame
  data_train_final = pd.DataFrame(curves_finals, dtype=float)

  # Interpolate missing values
  data_train_final = data_train_final.interpolate(axis=1)

  # Add labels and kepids
  data_train_final['kepid'] = pd.Series(kepids_finals)
  data_train_final['label'] = pd.Series(labels_finals)

  return data_train_final


def preprocess_test_data(path_test):
  """
    Load and preprocess Kepler light curve data for testing.

    This function:
    - Loads a CSV file with Kepler Object of Interest (KOI) test data.
    - Keeps only relevant columns for light curve processing.
    - Cleans NaN values and resets the index.
    - Downloads and preprocesses the light curves:
        - Removes NaNs and outliers
        - Normalizes flux
        - Folds light curves based on period and transit time
        - Masks transits and flattens the light curve
        - Stitches multiple curves and bins them
        - Standardizes the flux values
    - Returns a DataFrame containing processed curves with corresponding kepid.

    Parameters:
    ----------
    path_test : str
        Path to the CSV file containing KOI test data.

    Returns:
    -------
    pd.DataFrame
        Processed light curve data with flux columns and kepid.
    """

  # Load test data from CSV
  data_test = pd.read_csv(path_test, sep = ",")

  # Keep only relevant columns
  data_test = data_test[['kepid','koi_period','koi_time0bk','koi_duration','koi_quarters']]

  # Drop rows with missing values and reset index
  data_test = data_test.dropna()
  data_test = data_test.reset_index(drop=True)

  # Extract individual columns as arrays
  kepids = data_test['kepid'].values
  periods = data_test['koi_period'].values
  t0s = data_test['koi_time0bk'].values
  durations = data_test['koi_duration'].values

  # Initialize lists to store processed data
  kepids_finals = []
  curves_finals = []
  labels_finals = []

  # Loop through each KOI to process its light curve
  for idx, (kepid, period, t0, duration) in enumerate(zip(kepids, periods, t0s, durations)):

      try:
          # Download light curve(s) from Kepler
          lcs = search_lightcurve(str(kepid), author='Kepler', cadence='long').download_all()

          if lcs is None:
            print(idx, "not downloaded")
            continue

          # Ensure lcs is a list of LightCurve objects
          if isinstance(lcs, LightCurveCollection):
              lcs = list(lcs)
          elif isinstance(lcs, LightCurve):
              lcs = [lcs]

          # Remove NaNs and outliers
          lcs = [lc.remove_nans() for lc in lcs]
          lcs = [lc.remove_outliers(sigma=3) for lc in lcs]

          # Normalize flux
          normalize_lcs = [lc.normalize() for lc in lcs]

          flattened_curves = []

          for lc in normalize_lcs:
            # Fold the light curve using the period and transit epoch
            temp_fold = lc.fold(period, epoch_time=t0)
            fractional_duration = (duration / 24.0) / period
            phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
            transit_mask = np.isin(lc.time.value, temp_fold.time_original.value[phase_mask])

            # Flatten the light curve while masking the transit
            lc_flat, trend_lc = lc.flatten(return_trend=True, mask=transit_mask)
            flattened_curves.append(lc_flat)

          # Stitch all flattened light curves into one
          lc_stitched = LightCurveCollection(flattened_curves).stitch()

          # Fold and bin the stitched curve
          lc_fold = lc_stitched.fold(period, epoch_time=t0)
          lc_final = lc_fold.bin(bins=50)

          # Standardize flux
          flux_mean = np.nanmean(lc_final.flux)
          flux_std = np.nanstd(lc_final.flux)

          if flux_std == 0 or np.isnan(flux_std):
              curve_final = np.array(lc_final.flux.value - flux_mean, dtype=float)
          else:
              curve_final = np.array((lc_final.flux.value - flux_mean) / flux_std, dtype=float)

          # Append processed curve and kepid
          curves_finals.append(curve_final)
          kepids_finals.append(kepid)

      except Exception as e:
          print(idx, kepid, e)

  # Convert processed curves to DataFrame
  data_test_final = pd.DataFrame(curves_finals, dtype=float)

  # Interpolate missing values
  data_test_final = data_test_final.interpolate(axis=1)

  # Add kepid column
  data_test_final['kepid'] = pd.Series(kepids_finals)

  return data_test_final


def apply_lightgbm_model(data_train_final, data_test_final):
  """
    Apply a LightGBM classifier to detect exoplanets from Kepler data using nested cross-validation
    with Bayesian hyperparameter optimization, and return evaluation metrics, the trained final model,
    and predictions on the test set.

    Parameters
    ----------
    data_train_final : pandas.DataFrame
        Training dataset including features and label. The last column is assumed to be the label,
        and 'kepid' is the unique identifier for each Kepler target.
    data_test_final : pandas.DataFrame
        Test dataset including features and 'kepid' column for identification.

    Returns
    -------
    metrics : dict
        Dictionary containing mean and standard deviation of accuracy, precision, recall, and F1-score
        across outer cross-validation folds.
    best_final_model : lightgbm.LGBMClassifier
        LightGBM model trained on the full training set using the best hyperparameters found.
    results_tests : pandas.DataFrame
        DataFrame with test set predictions including 'kepid', predicted label, and probability of being an exoplanet.
    """

  # Make a copy of the training data to avoid modifying the original
  data_input = data_train_final.copy()

  # Extract feature names by dropping the label and identifier columns
  feature_names = data_train_final.drop(columns=[data_train_final.columns[-1], 'kepid']).columns

  # Extract labels
  label = data_input[data_input.columns[-1]]
  y = label.values

  # Extract features as numpy array
  X = data_input[feature_names].values

  # Binarize labels (0/1)
  lb = LabelBinarizer()
  y = lb.fit_transform(label)
  y = y.reshape(-1)

  # Define the hyperparameter search space for LightGBM
  param_lightgbm = {
      'learning_rate': Real(1e-3, 1, prior='log-uniform'),
      'n_estimators': Integer(10, 500),
      'num_leaves': Integer(2, 100),
      'max_depth': Integer(3, 10)
  }

  # Outer CV for model evaluation
  cv_outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

  # Lists to store CV results
  accuracies = []
  precisions = []
  recalls = []
  f1_scores = []

  # Loop over the outer cross-validation folds
  # cv_outer.split(X, y) generates indices for training and validation sets in each fold
  for fold, (train_ix, validation_ix) in enumerate(cv_outer.split(X, y)):

      X_train, X_validation = X[train_ix, :], X[validation_ix, :]
      y_train, y_validation = y[train_ix], y[validation_ix]

      # - X_train and y_train will be used to fit the model (and inner CV for hyperparameter tuning)
      # - X_validation and y_validation will be used to evaluate the model's performance
      #   on unseen data for this fold

      # Inner CV for Bayesian hyperparameter optimization
      cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

      base_model = LGBMClassifier(random_state=1)

      search = BayesSearchCV(
          estimator=base_model,
          search_spaces=param_lightgbm,
          scoring='accuracy',
          cv=cv_inner,
          n_iter=5, # number of hyperparameter combinations to try
          refit=True,
          random_state=1,
          n_jobs=-1
      )

      # Fit search to training data
      result = search.fit(X_train, y_train)

      # Get the best estimator from the inner CV
      best_model = result.best_estimator_

      # Predict on validation fold
      y_validation_predict = best_model.predict(X_validation)

      # Compute evaluation metrics
      acc = accuracy_score(y_validation, y_validation_predict)
      prec = precision_score(y_validation, y_validation_predict, average='binary', zero_division=1)
      rec = recall_score(y_validation, y_validation_predict, average='binary')
      f1 = f1_score(y_validation, y_validation_predict, average='binary')

      # Store metrics
      accuracies.append(acc)
      precisions.append(prec)
      recalls.append(rec)
      f1_scores.append(f1)

  # Aggregate metrics across folds
  metrics = {
          'accuracy_mean': np.mean(accuracies),
          'accuracy_std': np.std(accuracies),
          'precision_mean': np.mean(precisions),
          'precision_std': np.std(precisions),
          'recall_mean': np.mean(recalls),
          'recall_std': np.std(recalls),
          'f1_mean': np.mean(f1_scores),
          'f1_std': np.std(f1_scores)
      }

  # Train final model on full training data using the best hyperparameters
  best_final_model = LGBMClassifier(random_state=1, **search.best_params_)
  best_final_model.fit(X, y)

  # Prepare test set features
  X_test = data_test_final[feature_names].values
  test_ids = data_test_final['kepid'].values

  # Make predictions on the test set
  y_test_predict = best_final_model.predict(X_test)
  y_test_prob = best_final_model.predict_proba(X_test)[:, 1] #probability of being exoplanet

  # Store results in a DataFrame
  results_tests = pd.DataFrame({
      'kepid': test_ids,
      'pred_label': y_test_predict,
      'prob_exoplanet': y_test_prob
  })

  return metrics, best_final_model, results_tests


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


data_train_final = preprocess_train_data(path_train='./data/koi_candidates.csv')
data_test_final = preprocess_test_data(path_test='./data/koi_cumulative.csv')
metrics, best_final_model, results_tests = apply_lightgbm_model(data_train_final, data_test_final)

print(metrics)
print(best_final_model)
print(results_tests)
