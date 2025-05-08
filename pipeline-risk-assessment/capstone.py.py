import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pickle  # For loading the trained model
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, LineString

# --- Streamlit App ---
def main():
    st.title("Pipeline Risk Assessment Tool")

    # File upload for the new pipeline route
    pipeline_file = st.file_uploader(
        "Upload new pipeline route (GeoJSON or Shapefile .zip)",
        type=["geojson", "zip"],
    )

    # Load trained model
    try:
        # Attempt to load the model using a relative path (for broader compatibility)
        model_path = "pipeline_risk_assessment_final_model.pkl"
        with open(model_path, "rb") as file:
            stacking_model = pickle.load(file)
    except FileNotFoundError:
        st.error(
            "Error: Trained model 'pipeline_risk_assessment_final_model.pkl' not found. Please ensure the file is in the correct location."
        )
        return  # Stop if the model can't be loaded

    # Process and predict if a file is uploaded
    if pipeline_file is not None:
        try:
            # Load the pipeline data
            if pipeline_file.name.endswith(".zip"):
                pipeline_gdf = gpd.read_file(pipeline_file)
            else:
                pipeline_gdf = gpd.read_file(pipeline_file)

            if pipeline_gdf.empty:
                st.warning("Uploaded file is empty.")
                return

            # Reproject to a suitable CRS for calculations (e.g., a local UTM zone)
            pipeline_gdf = pipeline_gdf.to_crs(epsg=32737)  # Example: UTM Zone 37S, adjust as needed

            # --- Helper Functions ---
            def segment_pipeline(pipeline_row):
                """Segments a pipeline geometry into individual line segments."""
                coords = list(pipeline_row.geometry.coords)
                segments = []
                for i in range(len(coords) - 1):
                    from_point = Point(coords[i])
                    to_point = Point(coords[i + 1])
                    segment = {
                        "pipeline_id": pipeline_row["pipeline_id"],
                        "geometry": LineString([from_point, to_point]),
                    }
                    segments.append(segment)
                return segments

            # 1. Segment the pipeline
            new_segments_gdf = gpd.GeoDataFrame(crs=pipeline_gdf.crs)
            for idx, row in pipeline_gdf.iterrows():
                segmented = segment_pipeline(row)
                new_segments_gdf = pd.concat(
                    [new_segments_gdf, gpd.GeoDataFrame(segmented, crs=pipeline_gdf.crs)],
                    ignore_index=True,
                )

            new_segments_gdf = new_segments_gdf.to_crs(epsg=4326)  # Convert to WGS84 for folium

            # Dummy processing (replace with your actual feature engineering and risk prediction logic)
            new_segments_gdf["predicted_risk"] = "Medium"  # Replace with actual prediction logic

            # 6. Display results
            st.subheader("Risk Assessment Results")
            high_risk_segments = new_segments_gdf[
                new_segments_gdf["predicted_risk"] == "High"
            ]
            if len(high_risk_segments) > 0:
                st.write("High-Risk Segments:")
                st.write(high_risk_segments[["pipeline_id", "geometry", "predicted_risk"]])
            else:
                st.write("No high-risk segments found.")

            # 7. Map Visualization (using folium)
            st.subheader("Pipeline Risk Map")
            # Get the bounds of the pipeline for centering the map
            bounds = new_segments_gdf.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            for index, row in new_segments_gdf.iterrows():
                color = "green"  # Default color
                if row["predicted_risk"] == "High":
                    color = "red"
                elif row["predicted_risk"] == "Medium":
                    color = "yellow"
                folium.GeoJson(
                    row["geometry"],
                    style_function=lambda feature: {"color": color, "weight": 2},
                ).add_to(m)
            st_folium(m, width=700, height=500)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()