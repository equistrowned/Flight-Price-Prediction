✈️ Flight Price Predictor


🚀 ML-Powered Real-Time Fare Estimation System

  This project uses regression models to estimate flight prices using historical flight data. It includes:
    Data preprocessing & feature engineering
    Model training and evaluation
    Hyperparameter tuning
    A simple web interface built with Streamlit

🌟 Problem Statement

  Flight prices fluctuate due to multiple hidden factors like demand, route complexity, timing, and airline preferences.
  Users often struggle to estimate whether a ticket is overpriced or reasonable.

💡 Solution

  This project builds a machine learning-powered system that predicts flight ticket prices based on key travel parameters.

  👉 Users input flight details
  👉 Model analyzes patterns learned from historical data
  👉 Instantly returns an estimated price

🧠 Key Highlights
  ⚡ Real-time predictions via Streamlit UI
  📊 Advanced feature engineering (time, route, duration)
  🌲 Optimized Random Forest model
  🎯 Strong performance vs baseline regression
  🧩 End-to-end ML pipeline (data → model → deployment)
🛠️ Tech Stack
  Layer	Tools Used
  Language	Python
  Data	Pandas, NumPy
  ML Models	Scikit-learn
  Frontend	Streamlit
  Storage	Pickle
⚙️ System Architecture
  Raw Dataset → Feature Engineering → Model Training → Hyperparameter Tuning → Model Saving → Streamlit App → Prediction
📊 Model Performance
  Model	MAE ↓	R² ↑
  Linear Regression	~2467	~0.42
  Random Forest	~643	High improvement 🚀
  
  ✔ Random Forest significantly outperformed baseline
  ✔ Tuned using RandomizedSearchCV

🎮 Demo Flow
  Select Airline ✈️
  Choose Source & Destination 🌍
  Enter Stops & Duration ⏱️
  Set Departure & Arrival Time 🕒
  Click Predict Price 💰

💻 Run Locally
  git clone https://github.com/your-username/flight-price-predictor.git
  cd flight-price-predictor

  pip install -r requirements.txt
  streamlit run app.py
  
🔥 Future Enhancements
  📈 SHAP-based model explainability
  🌐 Live flight API integration
  🧠 Deep learning experimentation
  📊 Dynamic pricing trend visualization
  📱 Mobile-friendly UI

👨‍💻 Author
  Animesh Pathak

⭐ Show Some Love
  If you like this project:
    ⭐ Star the repo
    🍴 Fork it
    🧠 Suggest improvements
