# OptimizationApp.py

# Import necessary libraries
import streamlit as st
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, value

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

profits = {
    ('TShirt', 'Galway'): 8, ('Jacket', 'Galway'): 35, ('Sneakers', 'Galway'): 30, ('Dress', 'Galway'): 20,
    ('TShirt', 'Cork'): 10, ('Jacket', 'Cork'): 40, ('Sneakers', 'Cork'): 32, ('Dress', 'Cork'): 25,
    ('TShirt', 'Kilkenny'): 6, ('Jacket', 'Kilkenny'): 30, ('Sneakers', 'Kilkenny'): 28, ('Dress', 'Kilkenny'): 18,
}
weights = {'TShirt': 0.3, 'Jacket': 2.0, 'Sneakers': 1.5, 'Dress': 0.5}
volumes = {'TShirt': 0.002, 'Jacket': 0.01, 'Sneakers': 0.007, 'Dress': 0.004}
spaces = {'TShirt': 1, 'Jacket': 3, 'Sneakers': 2, 'Dress': 1}
distances = {'Galway': 186, 'Cork': 220, 'Kilkenny': 102}

# --- Optimization Model ---
model = LpProblem("Inventory_and_Routing_Optimization", LpMaximize)

x = LpVariable.dicts("Select", [(i, e) for i in items for e in events], 0, 1, LpBinary)
y = LpVariable.dicts("Visit", events, 0, 1, LpBinary)

model += (
    lpSum(profits[(i, e)] * x[(i, e)] for i in items for e in events) -
    cost_per_km * lpSum(distances[e] * y[e] for e in events)
)

model += lpSum(weights[i] * x[(i, e)] for i in items for e in events) <= van_weight, "WeightCapacity"
model += lpSum(volumes[i] * x[(i, e)] for i in items for e in events) <= van_volume, "VolumeCapacity"

for e in events:
    model += lpSum(spaces[i] * x[(i, e)] for i in items) <= display_capacity[e], f"DisplayCap_{e}"
    for i in items:
        model += x[(i, e)] <= y[e], f"LinkItemToVisit_{i}_{e}"

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
    st.markdown("## 📦 Product Allocation")
    for e in events:
        if y[e].value() == 1:
            st.markdown(f"**{e}**")
            table = ""
            for i in items:
                if x[(i, e)].value() == 1:
                    table += f"- {i}  \n"
            st.markdown(table or "_No items selected_")
