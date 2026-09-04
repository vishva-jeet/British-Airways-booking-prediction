from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# APP

app = FastAPI(title="British Airways Booking Prediction API")


# LOAD MODEL 

model = joblib.load("BA_model.pkl")
scaler = joblib.load("BA_scaler.pkl")
feature_columns = joblib.load("BA_feature_columns.pkl")

route_map = joblib.load("route_encoding_map.pkl")
origin_map = joblib.load("origin_encoding_map.pkl")
global_mean = joblib.load("global_target_mean.pkl")


# INPUT SCHEMA

class BookingInput(BaseModel):
    num_passengers: int
    sales_channel: str
    trip_type: str
    purchase_lead: int
    length_of_stay: int
    flight_hour: int
    flight_duration: float
    is_weekend: int
    wants_extra_baggage: int
    wants_preferred_seat: int
    wants_in_flight_meals: int
    booking_origin: str
    route: str


# ROUTES

@app.get("/")
def home():
    return {"message": "British Airways Booking Prediction API is running"}


@app.post("/predict")
def predict(data: BookingInput):

    try:


        input_df = pd.DataFrame({

            "num_passengers": [data.num_passengers],

            "sales_channel": [data.sales_channel],

            "trip_type": [data.trip_type],

            "purchase_lead": [data.purchase_lead],

            "length_of_stay": [data.length_of_stay],

            "flight_hour": [data.flight_hour],

            "flight_duration": [data.flight_duration],

            "is_weekend": [data.is_weekend],

            "wants_extra_baggage": [data.wants_extra_baggage],

            "wants_preferred_seat": [data.wants_preferred_seat],

            "wants_in_flight_meals": [data.wants_in_flight_meals],

            "booking_origin": [data.booking_origin],

            "route": [data.route]
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


        # Flight day number is not directly available in the simplified input, therefore use weekend information.
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


        # DROP RAW CATEGORICAL COLUMNS

        input_df = input_df.drop(
            columns=["route", "booking_origin"]
        )


        # ONE HOT ENCODING

        input_df = pd.get_dummies(
            input_df,
            columns=["sales_channel", "trip_type"],
            drop_first=True
        )


        # ALIGN WITH TRAINING FEATURES

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # SCALE

        input_scaled = scaler.transform(input_df)


        # PROBABILITY

        probability = model.predict_proba(
            input_scaled
        )[0][1]


        # Use your optimized threshold
        threshold = 0.52

        prediction = int(
            probability >= threshold
        )


        return {
            "probability": float(probability),
            "prediction": prediction,
            "status": "Likely to Book" if prediction == 1 else "Unlikely to Book",
            "threshold": threshold
        }

    except Exception as e:

        return {"error": str(e)}