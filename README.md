# Smart Agriculture System - Production Ready

## Version 2.0.0

An AI-powered smart agriculture platform for modern farming decisions. This system uses Machine Learning, Deep Learning, and Real-time APIs to help farmers make data-driven decisions.

---

## Features

### 1. Crop Prediction
- Predict the best crop for your field based on soil nutrients and climate
- **Technology**: Random Forest Classifier (ML)
- **Inputs**: N, P, K, Temperature, Humidity, pH, Rainfall
- **Output**: Top recommendations with confidence scores

### 2. Disease Detection
- Identify plant diseases from leaf images instantly
- **Technology**: CNN with MobileNetV2 (Deep Learning)
- **Inputs**: Leaf image (JPG, PNG, BMP, WebP)
- **Output**: Disease name with treatment advice

### 3. Growth Monitoring
- Track crop growth progress over time
- **Features**: Interactive charts, statistics, data logging
- **Export**: CSV data download

### 4. Market Analysis
- Analyze crop market prices and trends
- **Data**: Updated seasonal prices for 12+ crops
- **Charts**: Line charts, bar comparison, seasonal heatmaps

### 5. Fertigation System
- Smart irrigation and fertilizer recommendations
- **Inputs**: Crop type, growth stage, soil moisture
- **Output**: Water amount, NPK fertilizer, urgency alerts

### 6. AI Chatbot
- Get instant answers to farming questions
- **Technology**: Google Gemini AI
- **Setup**: Requires free API key (instructions below)

### 7. Real-time Weather
- Current weather data displayed in sidebar
- **Data**: Temperature, humidity, precipitation, wind speed
- **API**: Open-Meteo (free, no key required)

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
cd smart_agriculture
streamlit run app.py
```

### 3. Access the App

Open browser: http://localhost:8501

---

## AI Chatbot Setup

### Get Your Free Gemini API Key:

1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

### Enter API Key in the App:

1. Go to "AI Chatbot" section in the sidebar
2. Paste your API key in the input field
3. Start chatting about agriculture!

---

## Project Structure

```
smart_agriculture/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── models/               # Trained ML/DL models
│   ├── crop_model.pkl    # Crop prediction model
│   ├── label_encoder.pkl # Label encoder
│   ├── disease_model.h5  # Disease detection CNN
│   └── disease_labels.json
├── datasets/             # Data files
│   ├── crop_recommendation.csv
│   ├── market_prices.csv
│   └── growth_data.csv
├── training/             # Training scripts
│   ├── train_crop_model.py
│   └── train_disease_model.py
└── utils/                # Utility modules
    ├── fertigation.py
    └── visualization.py
```

---

## Technologies Used

- **Frontend**: Streamlit (Python web app)
- **ML**: Scikit-learn (Random Forest)
- **DL**: TensorFlow/Keras (MobileNetV2 CNN)
- **Data**: Pandas, NumPy
- **Visualization**: Plotly
- **AI**: Google Gemini API
- **Weather**: Open-Meteo API
- **Image Processing**: Pillow

---

## Features in Detail

### Responsive Design
- Works on desktop, tablet, and mobile
- Adapts to all screen sizes
- Touch-friendly interface

### Error Handling
- Validates all inputs
- Shows helpful error messages
- Graceful degradation

### Real-time Data
- Weather data from Open-Meteo API
- Seasonal price adjustments
- No manual updates needed

---

## Troubleshooting

### Model Not Found
```bash
python training/train_crop_model.py
python training/train_disease_model.py
```

### API Key Issues
- Make sure API key is correct
- Check internet connection
- Key should start with "AIza"

### Image Upload Fails
- Supported formats: JPG, PNG, BMP, WebP
- Minimum resolution: 50x50 pixels
- Use clear, well-lit images

---

## Future Enhancements

- Mobile app version
- IoT sensor integration
- Weather forecast integration
- Multi-language support
- SMS/Email notifications

---

## License

This is a Final Year Project.

---

## Support

For questions or issues, please refer to the documentation or contact the development team.

---

**Built with ❤️ for Farmers**
