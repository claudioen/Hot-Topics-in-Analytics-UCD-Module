# OptimizationApp.py

# Import necessary libraries
import streamlit as st
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, value

# --- UI CONFIG ---
st.set_page_config(page_title="Pop-Up Shop Optimizer", layout="centered")
st.title("📦 Pop-Up Shop Inventory & Routing Optimizer")
st.markdown("Optimize what to deliver and where — maximize profit, minimize route cost.")

# --- INPUTS ---
st.sidebar.header("🚚 Van Configuration")
van_weight = st.sidebar.slider("Van Weight Capacity (kg)", 20, 100, 50)
van_volume = st.sidebar.slider("Van Volume Capacity (m³)", 0.1, 1.0, 0.4)
travel_cost_per_km = st.sidebar.slider("Cost per km (€)", 0.1, 2.0, 0.5)

st.sidebar.header("🧺 Event Display Space")
display_caps = {
    'Galway': st.sidebar.slider("Galway Slots", 5, 30, 20),
    'Cork': st.sidebar.slider("Cork Slots", 5, 30, 25),
    'Kilkenny': st.sidebar.slider("Kilkenny Slots", 5, 30, 15)
}

# --- DATA ---
events = ['Galway', 'Cork', 'Kilkenny']
items = ['TShirt', 'Jacket', 'Sneakers', 'Dress']
profits = {
    ('TShirt', 'Galway'): 8,  ('Jacket', 'Galway'): 35, ('Sneakers', 'Galway'): 30, ('Dress', 'Galway'): 20,
    ('TShirt', 'Cork'): 10,  ('Jacket', 'Cork'): 40, ('Sneakers', 'Cork'): 32, ('Dress', 'Cork'): 25,
    ('TShirt', 'Kilkenny'): 6,  ('Jacket', 'Kilkenny'): 30, ('Sneakers', 'Kilkenny'): 28, ('Dress', 'Kilkenny'): 18,
}
weights = {'TShirt': 0.3, 'Jacket': 2.0, 'Sneakers': 1.5, 'Dress': 0.5}
volumes = {'TShirt': 0.002, 'Jacket': 0.01, 'Sneakers': 0.007, 'Dress': 0.004}
spaces = {'TShirt': 1, 'Jacket': 3, 'Sneakers': 2, 'Dress': 1}
distances = {'Galway': 186, 'Cork': 220, 'Kilkenny': 102}

# --- OPTIMIZATION ---
model = LpProblem("Inventory_Routing_Optimization", LpMaximize)
x = LpVariable.dicts("Select", [(i, e) for i in items for e in events], 0, 1, LpBinary)
y = LpVariable.dicts("Visit", events, 0, 1, LpBinary)

model += (
    lpSum(profits[(i, e)] * x[(i, e)] for i in items for e in events) -
    travel_cost_per_km * lpSum(distances[e] * y[e] for e in events)
)

model += lpSum(weights[i] * x[(i, e)] for i in items for e in events) <= van_weight
model += lpSum(volumes[i] * x[(i, e)] for i in items for e in events) <= van_volume

for e in events:
    model += lpSum(spaces[i] * x[(i, e)] for i in items) <= display_caps[e]
    for i in items:
        model += x[(i, e)] <= y[e]

model.solve()

# --- OUTPUT ---
st.subheader("🧮 Optimization Results")
st.success(f"**Net Profit:** €{value(model.objective):.2f}")

st.markdown("### ✅ Events Visited")
for e in events:
    if y[e].value() == 1:
        st.markdown(f"- {e} (Distance: {distances[e]} km)")

st.markdown("### 👕 Items Selected per Event")
for e in events:
    st.markdown(f"**{e}**")
    selected = [i for i in items if x[(i, e)].value() == 1]
    if selected:
        for i in selected:
            st.markdown(f"- {i}")
    else:
        st.markdown("_No items selected_")
