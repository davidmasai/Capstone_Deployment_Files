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
import os
import warnings
warnings.filterwarnings("ignore")

# --- Helper Functions ---
def segment_pipeline(pipeline_row, pipeline_id_col):  # Added pipeline_id_col
    """Segments a pipeline geometry into individual line segments."""
    coords = list(pipeline_row.geometry.coords)
    segments = []
    for i in range(len(coords) - 1):
        from_point = Point(coords[i])
        to_point = Point(coords[i + 1])
        segment = {
            "pipeline_id": pipeline_row[pipeline_id_col],  # Use the correct column
            "geometry": LineString([from_point, to_point]),
        }
        segments.append(segment)
    return segments

def calculate_curvature(gdf):
    """Calculates curvature metrics for pipeline segments."""
    gdf["segment_length"] = gdf.geometry.length
    gdf["angle_change"] = gdf.geometry.apply(calculate_angle_change)
    gdf["angle_segments"] = gdf["angle_change"] / gdf["segment_length"]
    gdf["mean_curvature"] = gdf["angle_segments"].abs()
    gdf["max_curvature"] = gdf["angle_segments"].abs()
    gdf["std_curvature"] = gdf["angle_segments"].abs()
    gdf["sharp_bend"] = (gdf["angle_change"].abs() > 30).astype(int)
    return gdf

def calculate_angle_change(line):
    """Calculates the angle change between consecutive line segments."""
    coords = list(line.coords)
    if len(coords) < 3:
        return 0.0
    angles = []
    for i in range(len(coords) - 2):
        p1 = np.array(coords[i])
        p2 = np.array(coords[i + 1])
        p3 = np.array(coords[i + 2])
        v1 = p2 - p1
        v2 = p3 - p2
        angle = np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        )
        angles.append(angle)
    return sum(angles)

def calculate_elevation_metrics(gdf, elevation_data):
    """Calculates elevation-related metrics for pipeline segments."""
    gdf['start_elevation'] = 0
    gdf['end_elevation'] = 0
    if 'start_elevation' in elevation_data and 'end_elevation' in elevation_data:
        gdf = gdf.merge(elevation_data[['geometry', 'start_elevation', 'end_elevation']], on='geometry', how='left')
    else:
        gdf['start_elevation'] = 0
        gdf['end_elevation'] = 0

    gdf["elevation_change"] = gdf["end_elevation"] - gdf["start_elevation"]
    gdf["gradient"] = gdf["elevation_change"] / gdf["segment_length"]
    gdf["gradient_percentage"] = gdf["gradient"] * 100
    gdf["gradient_degrees"] = np.degrees(np.arctan(gdf["gradient"]))
    gdf["steepness"] = pd.cut(
        gdf["gradient_degrees"],
        bins=[-90, -15, -5, 5, 15, 90],
        labels=["Very Steep Downhill", "Steep Downhill", "Moderate", "Steep Uphill", "Very Steep Uphill"],
    )
    return gdf

def integrate_waterways_with_pipelines(gdf, waterways_gdf):
    """Integrates pipeline segments with waterway data."""
    if waterways_gdf.empty:
        gdf["dist_to_waterway"] = 10000
        gdf["within_buffer"] = 0
        return gdf

    waterways_gdf = waterways_gdf.to_crs(gdf.crs)
    waterway_buffer = waterways_gdf.buffer(50)
    gdf["dist_to_waterway"] = gdf.geometry.apply(
        lambda geom: waterways_gdf.distance(geom).min()
    )
    gdf["within_buffer"] = gdf.geometry.intersects(waterway_buffer).astype(int)
    return gdf

def integrate_ecological_risks(gdf, lakes_gdf, forest_reserves_gdf, ecological_gdf, conservancy_gdf):
    """Integrates pipeline segments with ecological risk data."""
    gdf_crs = gdf.crs
    if not lakes_gdf.empty:
        lakes_gdf = lakes_gdf.to_crs(gdf_crs)
    if not forest_reserves_gdf.empty:
        forest_reserves_gdf = forest_reserves_gdf.to_crs(gdf_crs)
    if not ecological_gdf.empty:
        ecological_gdf = ecological_gdf.to_crs(gdf_crs)
    if not conservancy_gdf.empty:
        conservancy_gdf = conservancy_gdf.to_crs(gdf_crs)

    lake_buffer = lakes_gdf.buffer(500)
    forest_buffer = forest_reserves_gdf.buffer(500)
    park_buffer = ecological_gdf.buffer(500)
    conservancy_buffer = conservancy_gdf.buffer(500)

    gdf["lake_intersect_count"] = gdf.geometry.apply(
        lambda geom: sum(lake_buffer.intersects(geom))
    )
    gdf["lake_dist"] = gdf.geometry.apply(
        lambda geom: lakes_gdf.distance(geom).min() if not lakes_gdf.empty else 10000
    )
    gdf["lake_within_buf"] = gdf.geometry.intersects(lake_buffer).astype(int)
    gdf["lake_compliant"] = 1

    gdf["forest_intersect_count"] = gdf.geometry.apply(
        lambda geom: sum(forest_buffer.intersects(geom))
    )
    gdf["forest_dist"] = gdf.geometry.apply(
        lambda geom: forest_reserves_gdf.distance(geom).min() if not forest_reserves_gdf.empty else 10000
    )
    gdf["forest_within_buf"] = gdf.geometry.intersects(forest_buffer).astype(int)
    gdf["forest_compliant"] = 1

    gdf["park_intersect_count"] = gdf.geometry.apply(
        lambda geom: sum(park_buffer.intersects(geom))
    )
    gdf["park_dist"] = gdf.geometry.apply(
        lambda geom: ecological_gdf.distance(geom).min() if not ecological_gdf.empty else 10000
    )
    gdf["park_within_buf"] = gdf.geometry.intersects(park_buffer).astype(int)
    gdf["park_compliant"] = 1

    gdf["conservancy_intersect_count"] = gdf.geometry.apply(
        lambda geom: sum(conservancy_buffer.intersects(geom))
    )
    gdf["conservancy_dist"] = gdf.geometry.apply(
        lambda geom: conservancy_gdf.distance(geom).min() if not conservancy_gdf.empty else 10000
    )
    gdf["conservancy_within_buf"] = gdf.geometry.intersects(
        conservancy_buffer
    ).astype(int)
    gdf["conservancy_compliant"] = 1
    return gdf

def integrate_road_risk(gdf, transport_gdf):
    """Integrates pipeline segments with road risk data."""
    if transport_gdf.empty:
        gdf["dist_to_road"] = 10000
        gdf["within_no_disturb"] = 0
        gdf["within_row"] = 0
        gdf["road_risk"] = "None"
        return gdf

    transport_gdf = transport_gdf.to_crs(gdf.crs)
    road_buffer_50 = transport_gdf.buffer(50)
    road_buffer_200 = transport_gdf.buffer(200)

    gdf["dist_to_road"] = gdf.geometry.apply(
        lambda geom: transport_gdf.distance(geom).min()
    )
    gdf["within_no_disturb"] = gdf.geometry.intersects(road_buffer_50).astype(int)
    gdf["within_row"] = gdf.geometry.intersects(road_buffer_200).astype(int)

    gdf["road_risk"] = "None"
    gdf.loc[gdf["within_no_disturb"] == 1, "road_risk"] = "High"
    gdf.loc[
        (gdf["within_no_disturb"] == 0) & (gdf["within_row"] == 1), "road_risk"] = "Medium"
    return gdf

def flag_population_risk(gdf, settlement_gdf):
    """Flags pipeline segments at risk from population centers."""
    if settlement_gdf.empty:
        gdf["population_risk_score"] = 0
        gdf["settlement_count"] = 0
        gdf["dist_to_pop"] = 10000
        gdf["place"] = "None"
        gdf["nearest_settlement_type"] = "None"
        gdf["population_risk"] = "None"
        return gdf

    settlement_gdf = settlement_gdf.to_crs(gdf.crs)
    pop_buffer_5km = settlement_gdf.buffer(5000)

    gdf["population_risk_score"] = gdf.geometry.apply(
        lambda geom: sum(
            settlement_gdf.geometry.intersects(geom)
            * settlement_gdf["Population"]
        )
    )
    gdf["settlement_count"] = gdf.geometry.apply(
        lambda geom: sum(pop_buffer_5km.intersects(geom))
    )
    gdf["dist_to_pop"] = gdf.geometry.apply(
        lambda geom: settlement_gdf.distance(geom).min()
    )
    gdf["place"] = "None"
    gdf["nearest_settlement_type"] = "None"

    for index, row in gdf.iterrows():
        nearest_idx = settlement_gdf.distance(row.geometry).idxmin()
        nearest_settlement = settlement_gdf.loc[nearest_idx]
        gdf.loc[index, "place"] = nearest_settlement["COUNTY"]
        gdf.loc[index, "nearest_settlement_type"] = nearest_settlement["Class"]

    gdf["population_risk"] = "None"
    gdf.loc[gdf["dist_to_pop"] < 1000, "population_risk"] = "High"
    gdf.loc[
        (gdf["dist_to_pop"] >= 1000) & (gdf["dist_to_pop"] < 2000),
        "population_risk",
    ] = "Medium"
    gdf.loc[gdf["dist_to_pop"] >= 2000, "population_risk"] = "Low"
    return gdf

def add_counties(gdf, counties_gdf):
    """Adds county information to pipeline segments."""
    if counties_gdf.empty:
        gdf["County"] = "None"
        return gdf
    counties_gdf = counties_gdf.to_crs(gdf.crs)
    gdf["County"] = gdf.geometry.apply(
        lambda geom: ", ".join(
            [
                county["COUNTY"]
                for county in counties_gdf.loc[
                    counties_gdf.intersects(geom)
                ].to_dict("records")
            ]
            or "None"
        )
    )
    return gdf

# --- Main Streamlit App ---
def main():
    st.title("Pipeline Risk Assessment Tool")

    # File upload for the new pipeline route
    pipeline_file = st.file_uploader(
        "Upload new pipeline route (GeoJSON or Shapefile .zip)",
        type=["geojson", "zip"],
    )

    # Load trained model
    model_path = 'stacked_risk_model.pkl'
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as file:
                stacking_model = pickle.load(file)
        except Exception as e:
            st.error(f"Error: Failed to load the trained model: {e}")
            return
    else:
        st.error(
            "Error: Trained model 'stacked_risk_model.pkl' not found. Please ensure the file is in the correct location."
        )
        return

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

            pipeline_gdf = pipeline_gdf.to_crs(epsg=32737)

            # Determine the correct ID column name
            pipeline_id_col = 'pipeline_id' if 'pipeline_id' in pipeline_gdf.columns else 'ID'  # Use 'ID' if 'pipeline_id' is not present

            # 1. Segment the pipeline
            new_segments_gdf = gpd.GeoDataFrame(columns=pipeline_gdf.columns, crs=pipeline_gdf.crs)
            for idx, row in pipeline_gdf.iterrows():
                segmented = segment_pipeline(row, pipeline_id_col)  # Pass the ID column name
                new_segments_gdf = pd.concat(
                    [new_segments_gdf, gpd.GeoDataFrame(segmented, crs=pipeline_gdf.crs)],
                    ignore_index=True,
                )

            new_segments_gdf = new_segments_gdf.to_crs(epsg=4326)

            # Load other datasets
            data_files = {
                "waterways": "waterways.geojson",
                "lakes": "lakes.geojson",
                "forest_reserves": "forest_reserves.geojson",
                "ecological": "ecological.geojson",
                "conservancy": "conservancy.geojson",
                "transport": "transport.geojson",
                "urban_settlements": "urban_settlements.geojson",
                "kenya_counties": "kenya_counties.geojson",
            }
            data_gdfs = {}
            for name, filename in data_files.items():
                if os.path.exists(filename):
                    try:
                        data_gdfs[name] = gpd.read_file(filename)
                    except Exception as e:
                        st.error(f"Error loading {name} data: {e}")
                        return
                else:
                    st.warning(f"Warning: Data file '{filename}' not found.  Some features may not be calculated.")
                    data_gdfs[name] = gpd.GeoDataFrame()  # Create an empty GeoDataFrame

            # 2. Apply Feature Engineering
            new_segments_gdf = calculate_curvature(new_segments_gdf)
            new_segments_gdf = calculate_elevation_metrics(new_segments_gdf, new_segments_gdf)
            new_segments_gdf = integrate_waterways_with_pipelines(
                new_segments_gdf, data_gdfs["waterways"]
            )
            new_segments_gdf = integrate_ecological_risks(
                new_segments_gdf,
                data_gdfs["lakes"],
                data_gdfs["forest_reserves"],
                data_gdfs["ecological"],
                data_gdfs["conservancy"],
            )
            new_segments_gdf = integrate_road_risk(new_segments_gdf, data_gdfs["transport"])
            new_segments_gdf = flag_population_risk(
                new_segments_gdf, data_gdfs["urban_settlements"]
            )
            new_segments_gdf = add_counties(new_segments_gdf, data_gdfs["kenya_counties"])

            # 3. Handle missing values
            group_columns = [
                "angle_segments",
                "mean_curvature",
                "max_curvature",
                "std_curvature",
            ]
            for column in group_columns:
                if column in new_segments_gdf.columns:
                    new_segments_gdf[column] = new_segments_gdf.groupby(pipeline_id_col)[  # Use the correct column
                        column
                    ].transform(lambda x: x.fillna(x.median()))

            columns_with_lists = [
                "intersecting_waterways",
                "lake_intersecting_ids",
                "forest_intersecting_ids",
                "park_intersecting_ids",
                "conservancy_intersecting_ids",
            ]
            for col in columns_with_lists:
                if col in new_segments_gdf.columns:
                    new_segments_gdf[col] = new_segments_gdf[col].apply(
                        lambda x: str(x) if isinstance(x, list) else x
                    )

            categorical_columns = [
                "pipe_location",
                "place",
                "within_buffer",
                "lake_within_buf",
                "forest_within_buf",
                "park_within_buf",
                "conservancy_within_buf",
                "within_no_disturb",
                "within_row",
                "settlement_compliant",
                "lake_compliant",
                "forest_compliant",
                "park_compliant",
                "conservancy_compliant",
                "steepness"
            ]
            for col in categorical_columns:
                if col in new_segments_gdf.columns:
                    new_segments_gdf[col] = new_segments_gdf[col].astype("category")

            # 4. Prepare data for prediction
            if 'end_elevation' in new_segments_gdf.columns and 'start_elevation' in new_segments_gdf.columns:
                new_segments_gdf["elevation_change"] = (
                    new_segments_gdf["end_elevation"] - new_segments_gdf["start_elevation"]
                )
            else:
                new_segments_gdf["elevation_change"] = 0

            if 'std_curvature' in new_segments_gdf.columns and 'mean_curvature' in new_segments_gdf.columns:
                new_segments_gdf["curvature_variability"] = (
                    new_segments_gdf["std_curvature"] / new_segments_gdf["mean_curvature"]
                )
            else:
                new_segments_gdf["curvature_variability"] = 0

            numeric_features = [
                "segment_distance",
                "angle_segments",
                "mean_curvature",
                "max_curvature",
                "std_curvature",
                "start_elevation",
                "end_elevation",
                "dist_to_waterway",
                "lake_dist",
                "forest_dist",
                "park_dist",
                "conservancy_dist",
                "dist_to_road",
                "dist_to_pop",
                "lake_intersect_count",
                "forest_intersect_count",
                "park_intersect_count",
                "conservancy_intersect_count",
                "settlement_count",
                "elevation_change",
                "curvature_variability"
            ]
            categorical_features = [
                "pipe_location",
                "place",
                "within_buffer",
                "lake_within_buf",
                "forest_within_buf",
                "park_within_buf",
                "conservancy_within_buf",
                "within_no_disturb",
                "within_row",
                "settlement_compliant",
                "lake_compliant",
                "forest_compliant",
                "park_compliant",
                "conservancy_compliant",
                "steepness"
            ]
            all_features = [f for f in numeric_features + categorical_features if f in new_segments_gdf.columns]
            X_new = new_segments_gdf[all_features]

            # 5. Predict risk using the loaded model
            try:
                X_new_processed = stacking_model.named_estimators_['rf'][0].transform(X_new)
                predicted_risks = stacking_model.predict(X_new_processed)
                new_segments_gdf["predicted_risk"] = predicted_risks
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                return

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

            # 7. Map Visualization
            st.subheader("Pipeline Risk Map")
            bounds = new_segments_gdf.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            for index, row in new_segments_gdf.iterrows():
                color = "green"
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
