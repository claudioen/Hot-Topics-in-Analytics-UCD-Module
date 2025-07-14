# OptimizationApp.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import folium
from streamlit_folium import st_folium

# ---------------------------
# 1. Train ML model
# ---------------------------
@st.cache_data
def get_historical_data():
    df = pd.DataFrame({
        "city": ["Dublin", "Cork", "Galway", "Limerick", "Kilkenny", "Waterford"],
        "past_attendance": [1100, 950, 800, 700, 500, 550],
        "weather_score": [8, 7, 6, 7, 6, 7],
        "actual_customers": [1200, 900, 850, 750, 520, 600],
        "travel_cost": [250, 320, 320, 260, 10, 120]
    })
    df["city_encoded"] = df["city"].astype("category").cat.codes
    return df

@st.cache_data
def get_future_events():
    df = pd.DataFrame({
        "city": ["Dublin", "Cork", "Galway", "Limerick", "Kilkenny", "Waterford"],
        "past_attendance": [1200, 900, 850, 750, 520, 600],
        "weather_score": [9, 7, 7, 6, 7, 6],
        "travel_cost": [250, 320, 320, 260, 10, 120]
    })
    df["city_encoded"] = df["city"].astype("category").cat.codes
    return df

@st.cache_resource
def get_trained_model():
    historical_data = get_historical_data()
    X = historical_data[["city_encoded", "past_attendance", "weather_score"]]
    y = historical_data["actual_customers"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@st.cache_data
def get_predicted_future_events():
    model = get_trained_model()
    future_events = get_future_events().copy()
    X_new = future_events[["city_encoded", "past_attendance", "weather_score"]]
    future_events["expected_customers"] = model.predict(X_new)
    return future_events

# ---------------------------
# 3. Item definitions
# ---------------------------
@st.cache_data
def get_base_items():
    return [
        {"name": "t-shirt", "weight": 0.3, "cost": 7.00, "sale": 20.00, "profit": 13.00, "popularity": 0.3},
        {"name": "denim shorts", "weight": 0.4, "cost": 10.00, "sale": 28.00, "profit": 18.00, "popularity": 0.25},
        {"name": "crop top", "weight": 0.25, "cost": 6.00, "sale": 22.00, "profit": 16.00, "popularity": 0.2},
        {"name": "festival hat", "weight": 0.2, "cost": 3.00, "sale": 18.00, "profit": 15.00, "popularity": 0.15},
        {"name": "sundress", "weight": 0.5, "cost": 12.00, "sale": 35.00, "profit": 23.00, "popularity": 0.1},
    ]

# ---------------------------
# 4. Bounded Knapsack with Slots
# ---------------------------
def bounded_knapsack(items, capacity_kg, max_slots):
    scale = 100
    capacity = int(capacity_kg * scale)

    expanded = []
    for item in items:
        for _ in range(item["stock"]):
            expanded.append({
                "name": item["name"],
                "weight": int(item["weight"] * scale),
                "profit": item["profit"]
            })

    dp = [[0] * (max_slots + 1) for _ in range(capacity + 1)]
    counts = [[{} for _ in range(max_slots + 1)] for _ in range(capacity + 1)]

    for item in expanded:
        w = item["weight"]
        p = item["profit"]
        for cap in range(capacity, w - 1, -1):
            for slot in range(max_slots, 0, -1):
                if dp[cap - w][slot - 1] + p > dp[cap][slot]:
                    dp[cap][slot] = dp[cap - w][slot - 1] + p
                    counts[cap][slot] = counts[cap - w][slot - 1].copy()
                    counts[cap][slot][item["name"]] = counts[cap][slot].get(item["name"], 0) + 1

    return dp[capacity][max_slots], counts[capacity][max_slots]

# ---------------------------
# 5. Per-City Planning
# ---------------------------
@st.cache_data(show_spinner=False)
def compute_results(future_events, base_items, van_capacity_kg, tent_item_slots, conversion_rate):
    results = []
    for _, row in future_events.iterrows():
        city = row["city"]
        travel_cost = float(row["travel_cost"])
        expected_customers = float(row["expected_customers"])
        estimated_sales = max(1, int(expected_customers * conversion_rate))

        # Create city-specific item list
        city_items = []
        total_stock_units = 0

        for item in base_items:
            raw_est = max(1, int(item["popularity"] * estimated_sales))
            city_items.append({**item, "stock": raw_est})
            total_stock_units += raw_est

        # Normalize to fit tent slots
        if total_stock_units > tent_item_slots:
            scale_factor = tent_item_slots / total_stock_units
            for i in range(len(city_items)):
                new_stock = max(1, int(city_items[i]["stock"] * scale_factor))
                city_items[i]["stock"] = new_stock

        # Run knapsack
        profit, item_plan = bounded_knapsack(city_items, van_capacity_kg, tent_item_slots)
        net_profit = profit - travel_cost

        results.append({
            "city": city,
            "expected_customers": int(expected_customers),
            "travel_cost": travel_cost,
            "gross_profit": profit,
            "net_profit": net_profit,
            "items": item_plan
        })
    return results

# ---------------------------
# 6. Pick Top 3 Cities by Net Profit
# ---------------------------
def get_top_cities(results, n=3):
    return sorted(results, key=lambda x: x["net_profit"], reverse=True)[:n]

# ---------------------------
# 7. Print Results (for CLI only)
# ---------------------------
def print_top_cities(top_cities):
    for result in top_cities:
        print(f"\nCity: {result['city']}")
        print(f"Expected Customers: {result['expected_customers']}")
        print(f"Travel Cost: €{result['travel_cost']:.2f}")
        print(f"Gross Profit: €{result['gross_profit']:.2f}")
        print(f"Net Profit: €{result['net_profit']:.2f}")
        print("Items to Pack:")
        for item, qty in result["items"].items():
            print(f"  - {item}: {qty} units")

# ---------------------------
# --- Streamlit App Wrapper ---
# ---------------------------

def get_city_coords():
    # Hardcoded coordinates for demo; replace with real data as needed
    return {
        "Dublin": (53.3498, -6.2603),
        "Cork": (51.8985, -8.4756),
        "Galway": (53.2707, -9.0568),
        "Limerick": (52.6638, -8.6267),
        "Kilkenny": (52.6541, -7.2448),
        "Waterford": (52.2593, -7.1101)
    }

def get_city_order():
    # Example journey order for the map
    return ["Dublin", "Cork", "Limerick", "Galway", "Kilkenny", "Waterford"]

@st.cache_data(show_spinner=False)
def make_folium_map(city_order, city_coords, show_arrows=True, highlight_city=None):
    # Center map on Ireland
    m = folium.Map(location=[53.0, -8.0], zoom_start=7, control_scale=True)
    # Draw path with arrows
    for i in range(len(city_order) - 1):
        start = city_coords[city_order[i]]
        end = city_coords[city_order[i+1]]
        folium.PolyLine(
            [start, end],
            color="blue",
            weight=5,
            opacity=0.7,
            tooltip=f"{city_order[i]} → {city_order[i+1]}"
        ).add_to(m)
        # Add arrow marker for direction
        if show_arrows:
            # Calculate arrow position (midpoint)
            mid_lat = (start[0] + end[0]) / 2
            mid_lon = (start[1] + end[1]) / 2
            # Use a marker with a custom icon for the arrow
            folium.Marker(
                location=(mid_lat, mid_lon),
                icon=folium.DivIcon(html=f"""
                    <div style="transform: rotate({get_bearing(start, end)}deg);">
                        <span style="font-size:24px;color:red;">&#8594;</span>
                    </div>
                """),
                tooltip="Direction"
            ).add_to(m)
    # Add city markers
    for city, coords in city_coords.items():
        folium.Marker(
            location=coords,
            popup=city,
            icon=folium.Icon(color="green" if city == highlight_city else "blue", icon="info-sign")
        ).add_to(m)
    return m

def get_bearing(start, end):
    # Returns bearing in degrees from start to end
    import math
    lat1, lon1 = map(math.radians, start)
    lat2, lon2 = map(math.radians, end)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    return (bearing + 360) % 360

def get_results_df(results):
    # Flatten results for DataFrame
    rows = []
    for r in results:
        row = r.copy()
        row.pop("items")
        rows.append(row)
    return pd.DataFrame(rows)

def get_items_df(results, city):
    # Return items DataFrame for a city
    for r in results:
        if r["city"] == city:
            items = r["items"]
            return pd.DataFrame([{"Item": k, "Units": v} for k, v in items.items()])
    return pd.DataFrame()

def main():
    st.set_page_config(page_title="Company Name - Event Planning Demo", layout="wide")
    st.title("Company Name: Event Planning & Optimization Demo")
    st.markdown(
        "This interactive dashboard demonstrates event planning, demand forecasting, and logistics optimization for multiple cities. "
        "Use the filters to explore city-level and global analyses, and view the optimized journey on the map."
    )

    # --- Sidebar Filters ---
    st.sidebar.header("Filters & Experimentation")
    city_coords = get_city_coords()
    city_order = get_city_order()
    base_items = get_base_items()
    future_events = get_predicted_future_events()

    # City/global selector
    city_list = list(city_coords.keys())
    view_mode = st.sidebar.radio("View Mode", ["Global Overview", "City Detail"])
    selected_city = None
    if view_mode == "City Detail":
        selected_city = st.sidebar.selectbox("Select City", city_list)

    # Conversion rate slider (live experiment)
    conv_rate = st.sidebar.slider("Conversion Rate (%)", min_value=10, max_value=100, value=60, step=5)
    # Van capacity slider
    van_cap = st.sidebar.slider("Van Capacity (kg)", min_value=50, max_value=200, value=100, step=10)
    # Tent slots slider
    tent_slots = st.sidebar.slider("Tent Item Slots", min_value=10, max_value=100, value=50, step=5)

    # --- Results (cached) ---
    results = compute_results(
        future_events,
        base_items,
        van_cap,
        tent_slots,
        conv_rate / 100.0
    )
    top_cities = get_top_cities(results, n=3)

    # --- Main UI: Tabs for City/Global ---
    tab1, tab2 = st.tabs(["Global Overview", "City Detail"])

    with tab1:
        st.subheader("Top 3 Cities by Net Profit")
        df = get_results_df(top_cities)
        st.dataframe(df.style.format({"travel_cost": "€{:.2f}", "gross_profit": "€{:.2f}", "net_profit": "€{:.2f}"}), use_container_width=True)

        st.subheader("Optimized Journey Map")
        # Only redraw map if parameters change
        m = make_folium_map(city_order, city_coords, show_arrows=True)
        st_folium(m, width=900, height=500)

        st.subheader("All Cities - Summary Table")
        all_df = get_results_df(results)
        st.dataframe(all_df.style.format({"travel_cost": "€{:.2f}", "gross_profit": "€{:.2f}", "net_profit": "€{:.2f}"}), use_container_width=True)

    with tab2:
        st.subheader("City Detail")
        city = selected_city if selected_city else city_list[0]
        st.markdown(f"### {city}")
        # City summary
        for r in results:
            if r["city"] == city:
                st.metric("Expected Customers", r["expected_customers"])
                st.metric("Travel Cost", f"€{r['travel_cost']:.2f}")
                st.metric("Gross Profit", f"€{r['gross_profit']:.2f}")
                st.metric("Net Profit", f"€{r['net_profit']:.2f}")
                break
        # City items
        st.markdown("#### Items to Pack")
        items_df = get_items_df(results, city)
        st.dataframe(items_df, use_container_width=True)
        # City map
        st.markdown("#### City Location on Map")
        m2 = folium.Map(location=city_coords[city], zoom_start=10, control_scale=True)
        folium.Marker(
            location=city_coords[city],
            popup=city,
            icon=folium.Icon(color="green", icon="star")
        ).add_to(m2)
        st_folium(m2, width=700, height=400)

    # --- Footer ---
    st.markdown("---")
    st.caption("Demo app for Company Name. Powered by Streamlit, scikit-learn, and folium. Ready for Streamlit Cloud deployment.")

if __name__ == "__main__":
    main()
