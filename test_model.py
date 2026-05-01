import numpy as np
import pandas as pd
import joblib
import keras

def load_artifacts():
    print("Loading model and artifacts...")
    model = keras.saving.load_model('ids_lstm_model_patched.keras')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, scaler, le

def predict_sample(model, scaler, le, sample_df):
    """
    Predicts whether a given network traffic sample is Normal or an Attack.
    
    Args:
        model: Loaded Keras LSTM model.
        scaler: Loaded MinMaxScaler.
        le: Loaded LabelEncoder.
        sample_df: A pandas DataFrame containing exactly one row with the required features.
    
    Returns:
        predicted_label (str): 'Normal' or 'Attack'
        probability (float): The probability score of the prediction.
    """
    # 1. Clean data: Replace '-' with 0 and fill NaNs as done during training
    sample_df = sample_df.replace('-', 0)
    for col in sample_df.columns:
        sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce').fillna(0)
    
    # 2. Scale features
    X = sample_df.astype('float32')
    
    # Suppress feature name warnings if dummy names are used
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_scaled = scaler.transform(X.values)
    
    # 3. Reshape for LSTM: (samples, time_steps, features)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # 4. Predict
    prediction_prob = model.predict(X_reshaped, verbose=0)
    result_idx = (prediction_prob > 0.5).astype(int).flatten()[0]
    
    # 5. Decode label
    predicted_label = le.inverse_transform([result_idx])[0]
    
    return predicted_label, prediction_prob[0][0]

if __name__ == "__main__":
    try:
        model, scaler, le = load_artifacts()
        
        # Determine the expected number of features from the model's input shape
        expected_features = model.input_shape[2]
        print(f"Model expects {expected_features} features.")
        
        # Create a dummy sample with zeros for testing purposes
        # In a real production scenario, replace this with actual parsed network flow data
        dummy_data = {f"feature_{i}": [0] for i in range(expected_features)} 
        sample_df = pd.DataFrame(dummy_data)
        
        print("\nRunning inference on a test sample...")
        label, prob = predict_sample(model, scaler, le, sample_df)
        
        print(f"=====================================")
        print(f"Prediction Result : {label}")
        print(f"Attack Probability: {prob:.4f}")
        print(f"=====================================")
        
        print("\nNote: To use real data, ensure your categorical features are already ordinal-encoded, ")
        print("and the DataFrame contains the exact 32 features used during training.")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please ensure that 'ids_lstm_model.keras', 'scaler.pkl', and 'label_encoder.pkl' are in the same directory.")
