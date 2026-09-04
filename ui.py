import streamlit as st
import pandas as pd
import numpy as np
import joblib


# PAGE CONFIG

st.set_page_config(
    page_title="British Airways | Booking Prediction",
    page_icon="✈️",
    layout="wide"
)


#  CSS

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

h1 {
    font-size: 2.2rem;
    font-weight: 700;
}

h2 {
    font-size: 1.5rem;
}

.card {
    background-color: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #e6e9ef;
    margin-bottom: 20px;
}

.result-card {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    background-color: white;
    border: 1px solid #e6e9ef;
}

.metric-title {
    font-size: 14px;
    color: #6b7280;
}

.metric-value {
    font-size: 27px;
    font-weight: 700;
}

.small-text {
    color: #6b7280;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# LOAD MODEL

@st.cache_resource
def load_artifacts():

    model = joblib.load("BA_model.pkl")
    scaler = joblib.load("BA_scaler.pkl")
    feature_columns = joblib.load("BA_feature_columns.pkl")

    return model, scaler, feature_columns


try:

    model, scaler, feature_columns = load_artifacts()

except Exception as e:

    st.error(
        "Model files could not be loaded. "
        "Make sure BA_random_forest.pkl, BA_scaler.pkl "
        "and BA_feature_columns.pkl are in the same folder."
    )

    st.stop()


# HEADER

st.title("✈️ British Airways")
st.subheader("Customer Ticket Buying Behaviour Prediction")

st.markdown(
    """
    Predict the likelihood that a customer will complete a flight booking
    based on travel behaviour, booking characteristics and customer preferences.
    """
)

st.divider()


# SIDEBAR

with st.sidebar:

    st.header("Model Information")

    st.write("**Model:** XGBoost")
    st.write("**scale_pos_weight:** 5.6")
    st.write("**Trees:** 500")
    st.write("**learning_rate:** 0.05")
    st.write("**Max depth:** 5")
    st.write("**eval_metrics:** logloss")

    st.divider()

    st.caption(
        "This application uses the trained machine-learning model "
        "developed during the analysis."
    )


# INPUT SECTION

st.header("Customer & Booking Details")

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


# CUSTOMER

with col1:

    st.markdown("### Customer")

    num_passengers = st.number_input(
        "Number of Passengers",
        min_value=1,
        max_value=9,
        value=1,
        step=1
    )

    sales_channel = st.selectbox(
        "Sales Channel",
        ["Internet", "Mobile"]
    )


# TRIP

with col2:

    st.markdown("### Trip")

    trip_type = st.selectbox(
        "Trip Type",
        ["RoundTrip", "OneWay", "CircleTrip"]
    )

    flight_duration = st.number_input(
        "Flight Duration (hours)",
        min_value=0.0,
        max_value=25.0,
        value=7.5,
        step=0.1
    )

    length_of_stay = st.number_input(
        "Length of Stay (days)",
        min_value=0,
        max_value=365,
        value=7,
        step=1
    )


# BOOKING

with col3:

    st.markdown("### Booking")

    purchase_lead = st.number_input(
        "Purchase Lead (days)",
        min_value=0,
        max_value=365,
        value=30,
        step=1
    )

    flight_hour = st.slider(
        "Flight Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    is_weekend = st.selectbox(
        "Weekend Booking",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


st.markdown("</div>", unsafe_allow_html=True)


# CUSTOMER PREFERENCES

st.header("Customer Preferences")

pref1, pref2, pref3 = st.columns(3)

with pref1:

    wants_extra_baggage = st.selectbox(
        "Extra Baggage",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with pref2:

    wants_preferred_seat = st.selectbox(
        "Preferred Seat",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with pref3:

    wants_in_flight_meals = st.selectbox(
        "In-flight Meal",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


st.divider()


# LOCATION / ROUTE

st.header("Route Information")

loc1, loc2 = st.columns(2)

with loc1:

    booking_origin = st.text_input(
        "Booking Origin",
        value="Australia"
    )

with loc2:

    route = st.text_input(
        "Route",
        value="AKL-SYD"
    )


st.divider()


# PREDICTION

if st.button(
    "Predict Booking Probability",
    type="primary",
    use_container_width=True
):

    try:

        # CREATE RAW INPUT

        input_df = pd.DataFrame({

            "num_passengers": [num_passengers],

            "sales_channel": [sales_channel],

            "trip_type": [trip_type],

            "purchase_lead": [purchase_lead],

            "length_of_stay": [length_of_stay],

            "flight_hour": [flight_hour],

            "flight_duration": [flight_duration],

            "is_weekend": [is_weekend],

            "wants_extra_baggage": [wants_extra_baggage],

            "wants_preferred_seat": [wants_preferred_seat],

            "wants_in_flight_meals": [wants_in_flight_meals],

            "booking_origin": [booking_origin],

            "route": [route]
        })


        # FEATURE ENGINEERING

        # Total service preference
        input_df["all_services_pref_customer"] = (
            input_df["wants_extra_baggage"]
            + input_df["wants_preferred_seat"]
            + input_df["wants_in_flight_meals"]
        )

        # Additional service indicator
        input_df["extra_service_add_on"] = (
            input_df["all_services_pref_customer"] > 0
        ).astype(int)


        # Stay groups
        def create_stay_group(days):

            if days <= 7:
                return "0-7 days"

            elif days <= 30:
                return "8-30 days"

            elif days <= 60:
                return "31-60 days"

            elif days <= 90:
                return "61-90 days"

            elif days <= 120:
                return "91-120 days"

            elif days <= 180:
                return "121-180 days"

            elif days <= 250:
                return "181-250 days"

            elif days <= 365:
                return "251-365 days"

            return "365+ days"


        input_df["stay_group"] = input_df[
            "length_of_stay"
        ].apply(create_stay_group)


        # Flight day number is not directly available in the simplified UI, therefore use weekend information.
        input_df["flight_day_num"] = input_df["is_weekend"]


        # Purchase lead grouping
        def create_lead_group(days):

            if days <= 7:
                return "0-7"

            elif days <= 30:
                return "8-30"

            elif days <= 60:
                return "31-60"

            elif days <= 120:
                return "61-120"

            return "121+"


        input_df["lead_group"] = input_df[
            "purchase_lead"
        ].apply(create_lead_group)


        # TARGET ENCODING

        # These maps MUST come from the training data.
        route_map = joblib.load("route_encoding_map.pkl")
        origin_map = joblib.load("origin_encoding_map.pkl")

        global_mean = joblib.load("global_target_mean.pkl")

        input_df["route_encoded"] = (
            input_df["route"]
            .map(route_map)
            .fillna(global_mean)
        )

        input_df["booking_origin_encoded"] = (
            input_df["booking_origin"]
            .map(origin_map)
            .fillna(global_mean)
        )


        # DROP CATEGORICAL COLUMNS

        input_df = input_df.drop(
            columns=["route", "booking_origin"]
        )


        # ONE HOT ENCODING

        input_df = pd.get_dummies(
            input_df,
            columns=["sales_channel", "trip_type"],
            drop_first=True
        )


        # TRAINING FEATURES

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # ====================================================
        # SCALE
        # ====================================================

        input_scaled = scaler.transform(input_df)


        # ====================================================
        # PROBABILITY
        # ====================================================

        probability = model.predict_proba(
            input_scaled
        )[0][1]


        # Use your optimized RF threshold
        threshold = 0.52

        prediction = int(
            probability >= threshold
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.header("Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)


        with result_col1:

            st.markdown(
                f"""
                <div class="result-card">

                <div class="metric-title">
                Booking Probability
                </div>

                <div class="metric-value">
                {probability:.1%}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with result_col2:

            if prediction == 1:

                status = "Likely to Book"

            else:

                status = "Unlikely to Book"


            st.markdown(
                f"""
                <div class="result-card">

                <div class="metric-title">
                Prediction
                </div>

                <div class="metric-value">
                {status}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with result_col3:

            st.markdown(
f"""
                <div class="result-card">

                <div class="metric-title">
                Decision Threshold
                </div>

                <div class="metric-value">
                {threshold:.0%}
                </div>

                </div>
                """,                
                unsafe_allow_html=True
            )


        # Probability 

        st.progress(
            float(probability)
        )


        if prediction == 1:

            st.success(
                "The model predicts that this customer is likely "
                "to complete the booking."
            )

        else:

            st.info(
                "The model predicts that this customer is unlikely "
                "to complete the booking."
            )


        # BUSINESS INTERPRETATION

        st.subheader("Business Interpretation")

        if prediction == 1:

            st.write(
                "This customer shows relatively strong booking "
                "completion probability. The airline could prioritize "
                "conversion-focused offers or personalized services."
            )

        else:

            st.write(
                "This customer has a lower predicted booking "
                "completion probability. Targeted incentives or "
                "follow-up engagement may help improve conversion."
            )


    except Exception as e:

        st.error(
            f"Prediction could not be completed: {e}"
        )

        st.info(
            "Check that the saved preprocessing artifacts "
            "match the exact training pipeline."
        )

# FOOTER

st.divider()

st.caption(
    "British Airways Customer Ticket Buying Behaviour Prediction | "
    "Machine Learning Project"
)