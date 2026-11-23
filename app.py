
# STEP 46: IMPORT LIBRARIES AND LOAD MODEL

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Air Quality Prediction System",
    page_icon="🌍",
    layout="wide"
)

# Load Saved ML Model
MODEL_PATH = "models/air_quality_model.pkl"

if os.path.exists(MODEL_PATH):

    model_artifacts = joblib.load(MODEL_PATH)

    model = model_artifacts["model"]
    preprocessor = model_artifacts["preprocessor"]
    feature_columns = model_artifacts["feature_columns"]

else:
    st.error("Model file not found! Please check models folder.")
    st.stop()

# Main Heading
st.title("🌍 Air Quality Prediction and Analysis System")

st.markdown(
    "### AI-Based Next-Day Air Quality Prediction Using Machine Learning"
)

st.success("Machine Learning Model Loaded Successfully!")

st.write("Welcome to the Air Quality Prediction Dashboard.")

#=============================================================================================================================
#============================================================================================================================
# STEP 48: SIDEBAR NAVIGATION

st.sidebar.title("🌍 Air Quality System")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Home",
        "Dashboard",
        "AQI Prediction",
        "City Analysis",
        "Pollution Analysis",
        "Weather Analysis",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Air Quality Prediction System\n\n"
    "Developed Using Machine Learning"
)

st.sidebar.caption("BCA Data Science | Minor Project")

#============================================================================================================================
#============================================================================================================================

# STEP 49: HOME PAGE

if page == "Home":

    st.header("🏠 Welcome to Air Quality Prediction System")

    st.markdown("""
    This project uses Machine Learning to analyze
    historical air quality data and predict next-day AQI.
    """)

    st.markdown("---")

    st.subheader("🎯 Project Objectives")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("📊 Air Quality Analysis")
        st.write("Analyze historical air pollution data.")

    with col2:
        st.success("🤖 AQI Prediction")
        st.write("Predict next-day Air Quality Index.")

    with col3:
        st.warning("🌍 City Analysis")
        st.write("Compare air quality across Indian cities.")

    st.markdown("---")

    st.subheader("🛠️ Technologies Used")

    tech1, tech2, tech3, tech4 = st.columns(4)

    tech1.metric("Language", "Python")
    tech2.metric("ML Algorithm", "Gradient Boosting")
    tech3.metric("Data Analysis", "Pandas")
    tech4.metric("Web Framework", "Streamlit")

    st.markdown("---")

    st.subheader("📌 Project Information")

    st.write("**Project Title:** Air Quality Prediction and Analysis System Using ML")
    st.write("**Prediction Type:** Next-Day AQI Forecasting")
    st.write("**Dataset:** Historical Indian Air Quality Data")
    st.write("**Machine Learning:** Regression")
    st.write("**Best Model:** Gradient Boosting Regressor")


#================================================================================================================================
#===============================================================================================================================

# STEP 50: INTERACTIVE DASHBOARD

if page == "Dashboard":

    st.header("📊 Air Quality Dashboard")

    # Load Original Dataset
    DATA_PATH = "merged_air_quality.xls"

    if os.path.exists(DATA_PATH):

        dashboard_df = pd.read_csv(DATA_PATH)

        dashboard_df["date"] = pd.to_datetime(
            dashboard_df["date"]
        )

        # Dashboard Date Information
        st.subheader("📌 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Records",
                f"{len(dashboard_df):,}"
            )

        with col2:
            st.metric(
                "Total Cities",
                dashboard_df["city"].nunique()
            )

        with col3:
            st.metric(
                "Average AQI",
                round(dashboard_df["india_aqi"].mean(), 2)
            )

        with col4:
            st.metric(
                "Maximum AQI",
                dashboard_df["india_aqi"].max()
            )

        st.markdown("---")

        # City Selection
        st.subheader("🏙️ City-wise AQI Analysis")

        selected_city = st.selectbox(
            "Select City",
            sorted(dashboard_df["city"].unique())
        )

        city_df = dashboard_df[
            dashboard_df["city"] == selected_city
        ].sort_values("date")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "City Average AQI",
                round(city_df["india_aqi"].mean(), 2)
            )

        with col2:
            st.metric(
                "Latest Available AQI",
                city_df.iloc[-1]["india_aqi"]
            )

        # AQI Trend Chart
        st.subheader("📈 AQI Trend")

        st.line_chart(
            city_df.set_index("date")["india_aqi"]
        )

        st.markdown("---")

        # AQI Category Distribution
        st.subheader("🌍 AQI Category Distribution")

        category_counts = dashboard_df[
            "india_aqi_category"
        ].value_counts()

        st.bar_chart(category_counts)

        st.success("Dashboard Loaded Successfully!")

    else:
        st.error("Dataset file not found! Check dataset folder.")


#===============================================================================================================================
#===============================================================================================================================

# STEP 51: NEXT-DAY AQI PREDICTION

if page == "AQI Prediction":

    st.header("🤖 Next-Day AQI Prediction")

    st.write(
        "Enter today's pollution and weather parameters "
        "to predict tomorrow's AQI."
    )

    st.markdown("---")

    # Load Dataset for City List
    prediction_df = pd.read_csv("merged_air_quality.xls")

    city_list = sorted(prediction_df["city"].unique())

    st.subheader("🏙️ Select City")

    selected_city = st.selectbox(
        "Choose Your City",
        city_list
    )

    st.markdown("---")

    
# POLLUTION INPUTS
st.subheader("🌫️ Pollution Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    india_aqi = st.number_input("Today's India AQI", min_value=0.0, max_value=500.0, value=100.0, key="pred_india_aqi")
    pm2_5_mean = st.number_input("PM2.5 Mean", min_value=0.0, value=30.0, key="pred_pm25_mean")
    pm2_5_max = st.number_input("PM2.5 Maximum", min_value=0.0, value=50.0, key="pred_pm25_max")
    pm10_mean = st.number_input("PM10 Mean", min_value=0.0, value=50.0, key="pred_pm10_mean")
    pm10_max = st.number_input("PM10 Maximum", min_value=0.0, value=80.0, key="pred_pm10_max")
    no2_mean = st.number_input("NO2 Mean", min_value=0.0, value=20.0, key="pred_no2_mean")
    so2_mean = st.number_input("SO2 Mean", min_value=0.0, value=10.0, key="pred_so2_mean")
    o3_8h_max = st.number_input("O3 8-Hour Maximum", min_value=0.0, value=30.0, key="pred_o3_max")

with col2:
    co_8h_max = st.number_input("CO 8-Hour Maximum", min_value=0.0, value=1.0, key="pred_co_max")
    dust_mean = st.number_input("Dust Mean", min_value=0.0, value=10.0, key="pred_dust_mean")
    aod_mean = st.number_input("AOD Mean", min_value=0.0, value=0.2, key="pred_aod_mean")
    temperature_2m_max = st.number_input("Maximum Temperature", value=32.0, key="pred_temp_max")
    temperature_2m_min = st.number_input("Minimum Temperature", value=22.0, key="pred_temp_min")
    temperature_2m_mean = st.number_input("Mean Temperature", value=27.0, key="pred_temp_mean")
    apparent_temperature_max = st.number_input("Apparent Temperature", value=34.0, key="pred_apparent_temp")
    precipitation_sum = st.number_input("Precipitation Sum", min_value=0.0, value=0.0, key="pred_precipitation")

with col3:
    precipitation_hours = st.number_input("Precipitation Hours", min_value=0.0, value=0.0, key="pred_precip_hours")
    relative_humidity_2m_mean = st.number_input("Relative Humidity", min_value=0.0, max_value=100.0, value=60.0, key="pred_humidity")
    wind_speed_10m_max = st.number_input("Maximum Wind Speed", min_value=0.0, value=10.0, key="pred_wind_speed")
    wind_direction_10m_dominant = st.number_input("Wind Direction", min_value=0.0, max_value=360.0, value=180.0, key="pred_wind_direction")
    shortwave_radiation_sum = st.number_input("Shortwave Radiation", min_value=0.0, value=15.0, key="pred_radiation")
    et0_fao_evapotranspiration = st.number_input("Evapotranspiration", min_value=0.0, value=4.0, key="pred_evapotranspiration")
    weather_code = st.number_input("Weather Code", min_value=0.0, value=0.0, key="pred_weather_code")


# DATE INPUT
st.subheader("📅 Date Information")

selected_date = st.date_input(
    "Select Today's Date",
    value=pd.Timestamp.today().date(),
    key="prediction_date"
)

# PREDICTION BUTTON
st.markdown("---")

if st.button("🔮 Predict Next-Day AQI", key="predict_aqi_button", use_container_width=True):

    prediction_date = pd.to_datetime(selected_date)

    input_data = {
        "city": selected_city,
        "india_aqi": india_aqi,
        "pm2_5_mean": pm2_5_mean,
        "pm2_5_max": pm2_5_max,
        "pm10_mean": pm10_mean,
        "pm10_max": pm10_max,
        "no2_mean": no2_mean,
        "so2_mean": so2_mean,
        "o3_8h_max": o3_8h_max,
        "co_8h_max": co_8h_max,
        "dust_mean": dust_mean,
        "aod_mean": aod_mean,
        "temperature_2m_max": temperature_2m_max,
        "temperature_2m_min": temperature_2m_min,
        "temperature_2m_mean": temperature_2m_mean,
        "apparent_temperature_max": apparent_temperature_max,
        "precipitation_sum": precipitation_sum,
        "precipitation_hours": precipitation_hours,
        "relative_humidity_2m_mean": relative_humidity_2m_mean,
        "wind_speed_10m_max": wind_speed_10m_max,
        "wind_direction_10m_dominant": wind_direction_10m_dominant,
        "shortwave_radiation_sum": shortwave_radiation_sum,
        "et0_fao_evapotranspiration": et0_fao_evapotranspiration,
        "weather_code": weather_code,
        "month": prediction_date.month,
        "day_of_week": prediction_date.dayofweek,
        "day_of_year": prediction_date.dayofyear,
        "year": prediction_date.year
    }

    app_input = pd.DataFrame([input_data])

    app_input = app_input.reindex(
        columns=preprocessor.feature_names_in_
    )

    transformed_input = preprocessor.transform(app_input)

    predicted_aqi = model.predict(transformed_input)[0]
    predicted_aqi = np.clip(predicted_aqi, 0, 500)

    if predicted_aqi <= 50:
        category = "Good"
    elif predicted_aqi <= 100:
        category = "Satisfactory"
    elif predicted_aqi <= 200:
        category = "Moderate"
    elif predicted_aqi <= 300:
        category = "Poor"
    elif predicted_aqi <= 400:
        category = "Very Poor"
    else:
        category = "Severe"

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Next-Day AQI", round(float(predicted_aqi), 2))

    with col2:
        st.metric("Predicted AQI Category", category)

    st.success("Next-Day AQI Prediction Completed Successfully!")

    st.info(
        "Note: This prediction uses manually entered values "
        "and historical-data-based ML model. It is not live AQI data."
    )


#==============================================================================================================================
#==============================================================================================================================

# STEP 53: CITY ANALYSIS PAGE

if page == "City Analysis":

    st.header("🏙️ City-wise Air Quality Analysis")

    DATA_PATH = "merged_air_quality.xls"

    city_df = pd.read_csv(DATA_PATH)
    city_df["date"] = pd.to_datetime(city_df["date"])

    st.markdown("---")

    # CITY SELECTION
    selected_city_analysis = st.selectbox(
        "Select City for Analysis",
        sorted(city_df["city"].dropna().unique()),
        key="city_analysis_filter"
    )

    # FILTER DATA
    filtered_city = city_df[
        city_df["city"] == selected_city_analysis
    ].copy()

    st.subheader(f"📊 {selected_city_analysis} AQI Statistics")

    # AQI METRICS
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Average AQI",
            round(filtered_city["india_aqi"].mean(), 2)
        )

    with col2:
        st.metric(
            "Maximum AQI",
            filtered_city["india_aqi"].max()
        )

    with col3:
        st.metric(
            "Minimum AQI",
            filtered_city["india_aqi"].min()
        )

    with col4:
        st.metric(
            "Total Records",
            len(filtered_city)
        )

    st.markdown("---")

    # AQI TREND
    st.subheader("📈 Daily AQI Trend")

    daily_data = filtered_city.sort_values("date")

    st.line_chart(
        daily_data.set_index("date")["india_aqi"]
    )

    st.markdown("---")

    # MONTHLY AQI
    st.subheader("📅 Monthly Average AQI")

    monthly_data = (
        filtered_city
        .set_index("date")["india_aqi"]
        .resample("ME")
        .mean()
    )

    st.bar_chart(monthly_data)

    st.markdown("---")

    # CATEGORY DISTRIBUTION
    st.subheader("🌫️ AQI Category Distribution")

    category_data = filtered_city[
        "india_aqi_category"
    ].value_counts()

    st.bar_chart(category_data)

    st.success("City Analysis Loaded Successfully!")


#===============================================================================================================================
#===============================================================================================================================

# STEP 54: POLLUTION ANALYSIS PAGE

if page == "Pollution Analysis":

    st.header("🌫️ Pollution Analysis")

    pollution_df = pd.read_csv("merged_air_quality.xls")
    pollution_df["date"] = pd.to_datetime(pollution_df["date"])

    st.markdown("---")

    # CITY FILTER
    selected_pollution_city = st.selectbox(
        "Select City",
        ["All Cities"] + sorted(pollution_df["city"].dropna().unique()),
        key="pollution_city_filter"
    )

    if selected_pollution_city != "All Cities":
        filtered_pollution = pollution_df[
            pollution_df["city"] == selected_pollution_city
        ].copy()
    else:
        filtered_pollution = pollution_df.copy()

    # POLLUTANT SELECTION
    st.subheader("📊 Select Pollutant")

    pollutant_options = {
        "PM2.5": "pm2_5_mean",
        "PM10": "pm10_mean",
        "NO2": "no2_mean",
        "SO2": "so2_mean",
        "O3": "o3_8h_max",
        "CO": "co_8h_max",
        "Dust": "dust_mean"
    }

    selected_pollutant = st.selectbox(
        "Choose Pollutant",
        list(pollutant_options.keys()),
        key="pollutant_selection"
    )

    pollutant_column = pollutant_options[selected_pollutant]

    st.markdown("---")

    # POLLUTANT METRICS
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Value",
            round(filtered_pollution[pollutant_column].mean(), 2)
        )

    with col2:
        st.metric(
            "Maximum Value",
            round(filtered_pollution[pollutant_column].max(), 2)
        )

    with col3:
        st.metric(
            "Minimum Value",
            round(filtered_pollution[pollutant_column].min(), 2)
        )

    st.markdown("---")

    # POLLUTANT TREND
    st.subheader(f"📈 {selected_pollutant} Trend")

    trend_data = (
        filtered_pollution
        .groupby("date")[pollutant_column]
        .mean()
        .sort_index()
    )

    st.line_chart(trend_data)

    st.markdown("---")

    # POLLUTANT COMPARISON
    st.subheader("📊 Average Pollutant Comparison")

    comparison_columns = list(pollutant_options.values())

    comparison_data = (
        filtered_pollution[comparison_columns]
        .mean()
    )

    comparison_data.index = list(pollutant_options.keys())

    st.bar_chart(comparison_data)

    st.success("Pollution Analysis Loaded Successfully!")


#================================================================================================================================
#================================================================================================================================

# STEP 55: WEATHER ANALYSIS PAGE

if page == "Weather Analysis":

    st.header("🌦️ Weather and Air Quality Analysis")

    weather_df = pd.read_csv("merged_air_quality.xls")
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    st.markdown("---")

    # CITY FILTER
    selected_weather_city = st.selectbox(
        "Select City",
        ["All Cities"] + sorted(weather_df["city"].dropna().unique()),
        key="weather_city_filter"
    )

    if selected_weather_city != "All Cities":
        filtered_weather = weather_df[
            weather_df["city"] == selected_weather_city
        ].copy()
    else:
        filtered_weather = weather_df.copy()

    st.subheader("📊 Weather Statistics")

    # WEATHER METRICS
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Average Temperature",
            f"{filtered_weather['temperature_2m_mean'].mean():.2f} °C"
        )

    with col2:
        st.metric(
            "Average Humidity",
            f"{filtered_weather['relative_humidity_2m_mean'].mean():.2f}%"
        )

    with col3:
        st.metric(
            "Average Wind Speed",
            f"{filtered_weather['wind_speed_10m_max'].mean():.2f}"
        )

    with col4:
        st.metric(
            "Average AQI",
            f"{filtered_weather['india_aqi'].mean():.2f}"
        )

    st.markdown("---")

    # TEMPERATURE VS AQI
    st.subheader("🌡️ Temperature vs AQI")

    st.scatter_chart(
        filtered_weather,
        x="temperature_2m_mean",
        y="india_aqi"
    )

    st.markdown("---")

    # HUMIDITY VS AQI
    st.subheader("💧 Humidity vs AQI")

    st.scatter_chart(
        filtered_weather,
        x="relative_humidity_2m_mean",
        y="india_aqi"
    )

    st.markdown("---")

    # WIND SPEED VS AQI
    st.subheader("💨 Wind Speed vs AQI")

    st.scatter_chart(
        filtered_weather,
        x="wind_speed_10m_max",
        y="india_aqi"
    )

    st.markdown("---")

    # WEATHER CODE ANALYSIS
    st.subheader("🌤️ Weather Code Analysis")

    weather_code_data = (
        filtered_weather
        .groupby("weather_code")["india_aqi"]
        .mean()
        .sort_index()
    )

    st.bar_chart(weather_code_data)

    st.success("Weather Analysis Loaded Successfully!")


#==============================================================================================================================
#==============================================================================================================================

# STEP 56: MODEL PERFORMANCE PAGE

if page == "Model Performance":

    st.header("🤖 Machine Learning Model Performance")

    st.markdown(
        "Compare different Machine Learning models "
        "used for Next-Day AQI Prediction."
    )

    st.markdown("---")

    # MODEL PERFORMANCE DATA
    performance_data = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting"
        ],
        "MAE": [
            21.15,
            24.12,
            20.82,
            20.77
        ],
        "RMSE": [
            32.21,
            39.09,
            31.94,
            31.80
        ],
        "R2 Score": [
            0.7860,
            0.6849,
            0.7895,
            0.7914
        ]
    })

    # MODEL COMPARISON TABLE
    st.subheader("📊 Model Comparison Table")

    st.dataframe(
        performance_data,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # BEST MODEL
    best_model = performance_data.loc[
        performance_data["R2 Score"].idxmax()
    ]

    st.subheader("🏆 Best Performing Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best Model", best_model["Model"])

    with col2:
        st.metric("MAE", best_model["MAE"])

    with col3:
        st.metric("R² Score", best_model["R2 Score"])

    st.success(
        "Gradient Boosting is the best performing model "
        "based on the evaluated R² score."
    )

    st.markdown("---")

    # MAE COMPARISON
    st.subheader("📉 Mean Absolute Error (MAE)")

    st.bar_chart(
        performance_data.set_index("Model")["MAE"]
    )

    st.markdown("---")

    # RMSE COMPARISON
    st.subheader("📉 Root Mean Squared Error (RMSE)")

    st.bar_chart(
        performance_data.set_index("Model")["RMSE"]
    )

    st.markdown("---")

    # R2 COMPARISON
    st.subheader("📈 R² Score Comparison")

    st.bar_chart(
        performance_data.set_index("Model")["R2 Score"]
    )

    st.markdown("---")

    # MODEL INTERPRETATION
    st.subheader("📝 Model Performance Interpretation")

    st.write("""
    - **MAE:** Measures the average absolute prediction error.
    - **RMSE:** Gives more weight to larger prediction errors.
    - **R² Score:** Measures how well the model explains variations
      in the target variable.
    """)

    st.info(
        "Lower MAE and RMSE indicate better performance. "
        "Higher R² indicates better model performance."
    )

    st.success("Model Performance Page Loaded Successfully!")


#=============================================================================================================================
#=============================================================================================================================

# STEP 57: ABOUT PROJECT PAGE

if page == "About Project":

    st.header("📘 About Our Project")

    st.markdown("---")

    # PROJECT INTRODUCTION
    st.subheader("🌍 Project Introduction")

    st.write("""
    Air Quality Prediction and Analysis System Using Machine Learning
    is a data science project designed to analyze historical air
    pollution data across Indian cities and predict next-day
    Air Quality Index (AQI).

    The system uses Machine Learning regression algorithms
    to understand patterns in air quality and forecast future AQI.
    """)

    st.markdown("---")

    # PROJECT OBJECTIVES
    st.subheader("🎯 Project Objectives")

    objectives = [
        "Analyze historical air quality data.",
        "Predict next-day Air Quality Index.",
        "Compare air quality across Indian cities.",
        "Analyze major air pollutants.",
        "Study the relationship between weather and AQI.",
        "Compare Machine Learning model performance."
    ]

    for objective in objectives:
        st.write("✔️", objective)

    st.markdown("---")

    # DATASET INFORMATION
    st.subheader("📊 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", "78,200")

    with col2:
        st.metric("Total Cities", "50")

    with col3:
        st.metric("Dataset Columns", "31")

    st.write("**Dataset Period:** August 2022 – November 2026")

    st.write("**Dataset Type:** Historical Air Quality and Weather Data")

    st.markdown("---")

    # MACHINE LEARNING ALGORITHMS
    st.subheader("🤖 Machine Learning Algorithms")

    algorithms = [
        "Linear Regression",
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "Gradient Boosting Regressor"
    ]

    for algorithm in algorithms:
        st.write("🔹", algorithm)

    st.markdown("---")


    # MACHINE LEARNING ALGORITHMS AND PERFORMANCE

    st.subheader("🤖 Machine Learning Algorithms & Performance")

    performance_data = pd.DataFrame({
        "Algorithm": [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "Gradient Boosting Regressor"
        ],
        "MAE": [21.15, 24.12, 20.82, 20.77],
        "RMSE": [32.21, 39.09, 31.94, 31.80],
        "R2 Score": ["78.60%", "68.49%", "78.95%", "79.14%"]
    })

    st.dataframe(
        performance_data,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("🏆 Best Performing Algorithm")

    st.success(
        "Gradient Boosting Regressor achieved the highest "
        "R² Score of 79.14%."
    )

    st.write("""
    **Evaluation Metrics:**

    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R² Score: Coefficient of Determination

    Higher R² Score and lower MAE and RMSE indicate better
    regression model performance.
    """)


    # TECHNOLOGIES USED
    st.subheader("🛠️ Technologies Used")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.info("🐍 Python")

    with tech_col2:
        st.info("📊 Pandas & NumPy")

    with tech_col3:
        st.info("🤖 Scikit-learn")

    tech_col4, tech_col5, tech_col6 = st.columns(3)

    with tech_col4:
        st.info("📈 Matplotlib")

    with tech_col5:
        st.info("🌐 Streamlit")

    with tech_col6:
        st.info("💾 Joblib")

    st.markdown("---")

    # PROJECT WORKFLOW
    st.subheader("🔄 Project Workflow")

    workflow = [
        "Dataset Collection",
        "Data Cleaning",
        "Exploratory Data Analysis",
        "Feature Engineering",
        "Data Preprocessing",
        "Model Training",
        "Model Evaluation",
        "Next-Day AQI Prediction",
        "Streamlit Web Application"
    ]

    for index, step in enumerate(workflow, start=1):
        st.write(f"**Step {index}:** {step}")

    st.markdown("---")

    # PROJECT LIMITATIONS
    st.subheader("⚠️ Project Limitations")

    st.write("""
    - Predictions depend on the quality of the input data.
    - The model is trained using historical data.
    - The application does not provide live AQI readings.
    - Prediction accuracy may vary across cities and dates.
    """)

    st.markdown("---")

    st.success("About Project Page Loaded Successfully!")

    st.caption(
        "BCA Data Science | Minor Project | "
        "Air Quality Prediction and Analysis System Using ML"
    )
