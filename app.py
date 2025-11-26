import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import pickle

# Page configuration
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    .prediction-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🔬 Skin Lesion Classification System")
st.markdown("""
This application uses deep learning to classify skin lesions from the HAM10000 dataset.
Upload an image of a skin lesion to get a prediction.
""")

# Load model and encoders
@st.cache_resource
def load_models():
    try:
        model = keras.models.load_model('skin_lesion_classifier_final.h5')
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('lesion_type_dict.pkl', 'rb') as f:
            lesion_type_dict = pickle.load(f)
        
        return model, label_encoder, lesion_type_dict
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Preprocess image
def preprocess_image(image, img_size=224):
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize
    img_resized = cv2.resize(img_array, (img_size, img_size))
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

# Make prediction
def predict_lesion(model, image, label_encoder, lesion_type_dict):
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    
    # Get predicted class
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Get full name
    full_name = lesion_type_dict.get(predicted_class, predicted_class)
    
    # Get all probabilities
    all_predictions = []
    for idx, prob in enumerate(predictions[0]):
        class_name = label_encoder.inverse_transform([idx])[0]
        full_class_name = lesion_type_dict.get(class_name, class_name)
        all_predictions.append({
            'class': class_name,
            'full_name': full_class_name,
            'probability': prob * 100
        })
    
    # Sort by probability
    all_predictions = sorted(all_predictions, key=lambda x: x['probability'], reverse=True)
    
    return predicted_class, full_name, confidence, all_predictions

# Main app
def main():
    # Load models
    with st.spinner('Loading model...'):
        model, label_encoder, lesion_type_dict = load_models()
    
    if model is None:
        st.error("Failed to load the model. Please ensure all required files are present.")
        st.info("Required files: skin_lesion_classifier_final.h5, label_encoder.pkl, lesion_type_dict.pkl")
        return
    
    st.success("Model loaded successfully!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About the Model")
        st.markdown("""
        This model classifies skin lesions into 7 categories:
        
        - **nv**: Melanocytic nevi
        - **mel**: Melanoma
        - **bkl**: Benign keratosis-like lesions
        - **bcc**: Basal cell carcinoma
        - **akiec**: Actinic keratoses
        - **vasc**: Vascular lesions
        - **df**: Dermatofibroma
        
        **Note**: This is for educational purposes only. 
        Always consult a healthcare professional for medical diagnosis.
        """)
        
        st.header("Instructions")
        st.markdown("""
        1. Upload an image of a skin lesion
        2. Wait for the model to process
        3. View the prediction results
        4. Check the confidence scores
        """)
    
    # File uploader
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image of a skin lesion",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the skin lesion"
    )
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Display image info
            st.info(f"Image size: {image.size[0]} x {image.size[1]} pixels")
        
        with col2:
            st.subheader("Prediction Results")
            
            # Make prediction button
            if st.button("🔍 Classify Lesion", type="primary"):
                with st.spinner('Analyzing image...'):
                    try:
                        # Make prediction
                        predicted_class, full_name, confidence, all_predictions = predict_lesion(
                            model, image, label_encoder, lesion_type_dict
                        )
                        
                        # Display main prediction
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>Predicted Lesion Type</h3>
                            <h2 style="color: #4CAF50;">{full_name}</h2>
                            <p><strong>Class:</strong> {predicted_class}</p>
                            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display confidence level
                        if confidence > 80:
                            st.success("High confidence prediction")
                        elif confidence > 60:
                            st.warning("Moderate confidence prediction")
                        else:
                            st.warning("Low confidence prediction - consider consulting a specialist")
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
        
        # Display all predictions
        if uploaded_file is not None and st.session_state.get('predictions_made', False):
            st.header("📊 Detailed Probability Distribution")
            
            # Get predictions again (or store in session state)
            predicted_class, full_name, confidence, all_predictions = predict_lesion(
                model, image, label_encoder, lesion_type_dict
            )
            
            # Create a dataframe for display
            import pandas as pd
            df_predictions = pd.DataFrame(all_predictions)
            df_predictions['probability'] = df_predictions['probability'].apply(lambda x: f"{x:.2f}%")
            df_predictions.columns = ['Class Code', 'Lesion Type', 'Probability']
            
            st.dataframe(df_predictions, use_container_width=True, hide_index=True)
            
            # Visualization
            st.subheader("Probability Chart")
            chart_data = pd.DataFrame({
                'Lesion Type': [p['full_name'][:30] for p in all_predictions],
                'Probability (%)': [p['probability'] for p in all_predictions]
            })
            st.bar_chart(chart_data.set_index('Lesion Type'))
        
        # Set session state
        if uploaded_file is not None and 'predictions_made' not in st.session_state:
            st.session_state.predictions_made = False
        
        if st.button("🔍 Classify Lesion", type="primary", key="classify_btn"):
            st.session_state.predictions_made = True
            st.rerun()
    
    else:
        st.info("👆 Please upload an image to get started")
        
        # Display sample information
        st.header("Sample Results")
        st.markdown("""
        Once you upload an image, you'll see:
        - The predicted lesion type
        - Confidence level of the prediction
        - Detailed probability distribution for all classes
        - Visual chart of probabilities
        """)

# Disclaimer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>⚠️ Medical Disclaimer</strong></p>
        <p>This tool is for educational and research purposes only. It should not be used as a substitute 
        for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician 
        or other qualified health provider with any questions you may have regarding a medical condition.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()