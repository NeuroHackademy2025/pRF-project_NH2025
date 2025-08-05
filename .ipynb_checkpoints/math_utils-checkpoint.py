def weighted_nan_median(data, weights):
    """
    Calculate the weighted median of a data array, ignoring NaN values.

    Parameters:
    data (pd.Series, pd.DataFrame, np.ndarray): Data points, may contain NaN values.
    weights (pd.Series, pd.DataFrame, np.ndarray): Weights corresponding to the data points.

    Returns:
    float: The weighted median of the data points, ignoring NaN values.
           Returns NaN if the cumulative weights are not defined.
    """
    import numpy as np
    import pandas as pd

    # Convert data and weights to pandas Series if they are numpy arrays
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    if isinstance(weights, np.ndarray):
        weights = pd.Series(weights)
        
    # If data and weights are DataFrames, ensure they have a single column
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("DataFrame data must have exactly one column")
        data = data.iloc[:, 0]
        
    if isinstance(weights, pd.DataFrame):
        if weights.shape[1] != 1:
            raise ValueError("DataFrame weights must have exactly one column")
        weights = weights.iloc[:, 0]

    # Mask NaN values in the data
    mask = ~data.isna()

    # Apply the mask to data and weights
    masked_data = data[mask].reset_index(drop=True)
    masked_weights = weights[mask].reset_index(drop=True)

    # Check if there are no valid data points
    if masked_data.size == 0 or masked_weights.size == 0:
        return np.nan

    # Sort the data and corresponding weights
    masked_data = pd.to_numeric(masked_data, errors='coerce').dropna()
    sorted_indices = np.argsort(masked_data)
    sorted_data = masked_data.iloc[sorted_indices].reset_index(drop=True)
    sorted_weights = masked_weights.iloc[sorted_indices].reset_index(drop=True)

    # Calculate the cumulative sum of weights
    cumulative_weights = np.cumsum(sorted_weights)

    # Check if cumulative_weights is defined and has elements
    if cumulative_weights.size == 0:
        return np.nan

    # Find the median position
    median_weight = cumulative_weights.iloc[-1] / 2.0

    # Find the index where the cumulative weight crosses the median weight
    median_index = np.searchsorted(cumulative_weights, median_weight)

    return sorted_data.iloc[median_index]

def weighted_nan_percentile(data, weights, percentile):
    """
    Calculate the weighted percentile of an array or a pandas Series, ignoring NaN values.

    Parameters:
    data (np.ndarray or pd.Series): Array or pandas Series of data points, may contain NaN values.
    weights (np.ndarray or pd.Series): Array or pandas Series of weights corresponding to the data points.
    percentile (float): Percentile to compute, between 0 and 100.

    Returns:
    float: The weighted percentile of the data points, ignoring NaN values.
           Returns NaN if the cumulative weights are not defined.
    """
    import numpy as np
    import pandas as pd
    
    # Convert pandas Series to numpy array if needed
    if isinstance(data, pd.Series):
        data = data.values
    if isinstance(weights, pd.Series):
        weights = weights.values
    
    # Mask NaN values in the data
    mask = ~np.isnan(data)
    
    # Apply the mask to data and weights
    masked_data = data[mask]
    masked_weights = weights[mask]
    
    # Check if there are no valid data points
    if masked_data.size == 0 or masked_weights.size == 0:
        return np.nan
    
    # Sort the data and corresponding weights
    sorted_indices = np.argsort(masked_data)
    sorted_data = masked_data[sorted_indices]
    sorted_weights = masked_weights[sorted_indices]
    
    # Calculate the cumulative sum of weights
    cumulative_weights = np.cumsum(sorted_weights)
    
    # Check if cumulative_weights is defined and has elements
    if cumulative_weights.size == 0:
        return np.nan
    
    # Calculate the percentile weight
    percentile_weight = percentile / 100.0 * cumulative_weights[-1]
    
    # Find the index where the cumulative weight crosses the percentile weight
    percentile_index = np.searchsorted(cumulative_weights, percentile_weight)
    
    return float(sorted_data[percentile_index])

def weighted_regression(x_reg, y_reg, weight_reg, model):
    """
    Function to compute regression parameter weighted by a matrix (e.g. r2 value),
    where the regression model is y = 1/(cx) + d.

    Parameters
    ----------
    x_reg : array (1D)
        x values to regress
    y_reg : array
        y values to regress
    weight_reg : array (1D) 
        weight values (0 to 1) for weighted regression
    model : str
        Type of regression model, either 'pcm' for the original model or 'linear' for a linear model.

    Returns
    -------
    coef_reg : float or array
        regression coefficient(s)
    intercept_reg : float or str
        regression intercept or a string indicating no intercept (for linear model)
    """

    import numpy as np
    from scipy.optimize import curve_fit
    from sklearn import linear_model
    
    
    
    x_reg = np.array(x_reg)
    y_reg = np.array(y_reg)
    
    weight_reg = np.array(weight_reg)

    # Filter out NaN values
    x_reg_nan = x_reg[(~np.isnan(x_reg) & ~np.isnan(y_reg))]
    y_reg_nan = y_reg[(~np.isnan(x_reg) & ~np.isnan(y_reg))]
    weight_reg_nan = weight_reg[~np.isnan(weight_reg)]
    
    if model == 'pcm':
        # Define the model function
        def model_function(x, c, d):
            return 1 / (c * x + d)

        if weight_reg_nan.size >= 2:
            # Perform curve fitting
            params, _ = curve_fit(model_function, x_reg_nan, y_reg_nan, sigma=weight_reg_nan, maxfev=10000)
            c, d = params
        else:
            c, d = np.nan, np.nan
        return c, d

    elif model == 'linear':
        if weight_reg_nan.size >= 2:
            regr = linear_model.LinearRegression()
            
            # Filter out NaN values
            x_reg_nan = x_reg_nan.reshape(-1, 1)
            y_reg_nan = y_reg_nan.reshape(-1, 1)
            
            regr.fit(x_reg_nan, y_reg_nan, sample_weight=weight_reg_nan)
            coef_reg, intercept_reg = regr.coef_[0][0], regr.intercept_[0]
        else: 
            coef_reg, intercept_reg = np.nan, np.nan
        return coef_reg, intercept_reg
    else:
        raise ValueError("Invalid model type. Supported models are 'pcm' and 'linear'.")

