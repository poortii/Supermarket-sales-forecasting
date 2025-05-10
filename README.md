# 🛒 Supermarket Sales Forecasting

This project is a **Streamlit dashboard** that performs **category-wise time series forecasting** using multiple models, including:

- **Prophet** (by Meta/Facebook)
- **Holt-Winters Exponential Smoothing**
- **LSTM (Recurrent Neural Network)**
- **XGBoost Regression**

📊 The app allows users to explore, visualize, and compare forecasts for different product categories (e.g., Furniture, Office Supplies, Technology) using real-world sales data from the **Superstore dataset**.

---

## 📦 Features

- 📈 Interactive forecasting of sales by category
- 📊 Comparison of predictions from different models
- 🧠 Evaluation metrics (MAE, RMSE, MAPE)
- 📉 Trend, seasonality, and future estimates
- 🖥️ Fully built with Python and Streamlit

---

## 📂 Project Structure

forecast-app/
├── app.py # Streamlit app entry point
├── forecast_utils.py # All forecasting models and utility functions
├── Superstore.xls # Raw dataset (Excel)
├── requirements.txt # List of Python dependencies
└── new_env_310/ # Python 3.10 virtual environment (ignored in Git)


---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/poortii/Supermarket-sales-forecasting.git
cd Supermarket-sales-forecasting

```
### 2. Create a virtual environment (Python 3.10 recommended)
python -m venv env
env\Scripts\activate  # On Windows
# OR
source env/bin/activate  # On macOS/Linux

### 3. Install required dependencies
pip install -r requirements.txt

### 4. Run the Streamlit app
streamlit run app.py

### 📊 Dataset
The dataset used is Sample - Superstore.xls which contains sales data categorized by:

- Category & Sub-Category

- Order Date

- Sales, Profit, Quantity, Region, etc.

### 📈 Models Used

Prophet	
Holt-Winters	
LSTM	
XGBoost	

### 💡 Future Ideas
Forecast by Region or Sub-Category

Model hyperparameter tuning

Include ARIMA or SARIMA for comparison

Deploy online via Streamlit Cloud or HuggingFace Spaces


### ✨ Author
Made with ❤️ by @poortii



