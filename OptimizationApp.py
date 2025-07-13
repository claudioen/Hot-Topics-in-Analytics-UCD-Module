# OptimizationApp.py

import folium
from streamlit_folium import st_folium

# Coordinates for events and depot (Dublin)
locations = {
    "Dublin": (53.3498, -6.2603),
    "Galway": (53.2707, -9.0568),
    "Cork": (51.8985, -8.4756),
    "Kilkenny": (52.6541, -7.2448)
}


# Import necessary libraries
import streamlit as st
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, LpInteger, value

st.set_page_config(page_title="📦 StyleSprint Optimizer", layout="wide")

# --- Sidebar Inputs ---
st.sidebar.title("⚙️ Configuration Panel")

st.sidebar.subheader("🚚 Van Settings")
van_weight = st.sidebar.slider("Max Weight (kg)", 20, 100, 50, step=5)
van_volume = st.sidebar.slider("Max Volume (m³)", 0.1, 1.0, 0.4, step=0.05)
cost_per_km = st.sidebar.slider("Fuel Cost per km (€)", 0.1, 2.0, 0.5, step=0.1)

st.sidebar.subheader("🧺 Display Capacity per Event")
display_capacity = {
    "Galway": st.sidebar.slider("Galway (slots)", 5, 30, 20),
    "Cork": st.sidebar.slider("Cork (slots)", 5, 30, 25),
    "Kilkenny": st.sidebar.slider("Kilkenny (slots)", 5, 30, 15)
}

# --- Data Definitions ---
events = ['Galway', 'Cork', 'Kilkenny']
items = ['TShirt', 'Jacket', 'Sneakers', 'Dress']

# ✅ Updated realistic parameters
weights = {
    'TShirt': 0.25,
    'Jacket': 1.8,
    'Sneakers': 1.2,
    'Dress': 0.6
}

volumes = {
    'TShirt': 0.0015,
    'Jacket': 0.008,
    'Sneakers': 0.006,
    'Dress': 0.0035
}

spaces = {
    'TShirt': 1,
    'Jacket': 3,
    'Sneakers': 2,
    'Dress': 1
}

profits = {
    ('TShirt', 'Galway'): 10,   ('Jacket', 'Galway'): 40,   ('Sneakers', 'Galway'): 28,   ('Dress', 'Galway'): 22,
    ('TShirt', 'Cork'): 12,     ('Jacket', 'Cork'): 42,     ('Sneakers', 'Cork'): 30,     ('Dress', 'Cork'): 25,
    ('TShirt', 'Kilkenny'): 9,  ('Jacket', 'Kilkenny'): 38, ('Sneakers', 'Kilkenny'): 26, ('Dress', 'Kilkenny'): 20
}

distances = {'Galway': 186, 'Cork': 220, 'Kilkenny': 102}

# --- Optimization Model ---
model = LpProblem("Inventory_and_Routing_Optimization", LpMaximize)

# Decision Variables
x = LpVariable.dicts("Select", [(i, e) for i in items for e in events], lowBound=0, cat=LpInteger)
y = LpVariable.dicts("Visit", events, 0, 1, LpBinary)

# Objective: Maximize profit - travel cost
model += (
    lpSum(profits[(i, e)] * x[(i, e)] for i in items for e in events) -
    cost_per_km * lpSum(distances[e] * y[e] for e in events)
)

# Capacity Constraints (global)
model += lpSum(weights[i] * x[(i, e)] for i in items for e in events) <= van_weight, "WeightCapacity"
model += lpSum(volumes[i] * x[(i, e)] for i in items for e in events) <= van_volume, "VolumeCapacity"

# Display and linking constraints
for e in events:
    model += lpSum(spaces[i] * x[(i, e)] for i in items) <= display_capacity[e], f"DisplayCap_{e}"
    for i in items:
        model += x[(i, e)] <= 1000 * y[e], f"LinkItemToVisit_{i}_{e}"

# Solve
model.solve()

# --- Layout: Two-column Results ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 📈 Optimization Summary")
    st.metric(label="Net Profit (€)", value=f"{value(model.objective):.2f}")
    st.markdown("### ✅ Events Selected")
    for e in events:
        if y[e].value() == 1:
            st.markdown(f"- {e} ({distances[e]} km)")

with col2:
    st.markdown("## 📦 Product Quantities per Event")
    for e in events:
        if y[e].value() == 1:
            st.markdown(f"**{e}**")
            table = ""
            for i in items:
                qty = x[(i, e)].value()
                if qty and qty > 0:
                    table += f"- {i}: **{int(qty)} units**  \n"
            st.markdown(table or "_No items selected_")



# --- Map Visualization ---
st.markdown("## 🗺️ Delivery Route Map")

# Filter visited locations
route = ['Dublin'] + [e for e in events if y[e].value() == 1] + ['Dublin']  # round trip

m = folium.Map(location=locations["Dublin"], zoom_start=7)

# Add markers
for loc in route:
    folium.Marker(
        locations[loc],
        popup=loc,
        icon=folium.Icon(color='blue' if loc != 'Dublin' else 'green')
    ).add_to(m)

# Draw route
coords = [locations[loc] for loc in route]
folium.PolyLine(coords, color="red", weight=2.5, opacity=0.9).add_to(m)

# Show map in Streamlit
st_folium(m, width=725, height=450)
