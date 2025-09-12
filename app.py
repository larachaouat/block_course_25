import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def logistic_step(N_prev, K, r, T):
    return K / (1 + (K / N_prev - 1) * np.exp(-r * T))

def simulate(N0, K, r, T, I, n_steps):
    times = [0]
    pops = [N0]
    N = N0
    for step in range(1, n_steps + 1):
        N = logistic_step(N, K, r, T)
        N = (1 - I) * N
        times.append(step * T)
        pops.append(N)
    return np.array(times), np.array(pops)

# --- Streamlit interface ---
st.title("📈 Logistic Growth with Periodic Removal")

K = st.slider("Carrying capacity K", 10, 500, 100, 10)
r = st.slider("Growth rate r", 0.01, 1.0, 0.3, 0.01)
T = st.slider("Period T", 0.1, 5.0, 1.0, 0.1)
I = st.slider("Removal fraction I (%)", 0, 90, 20, 5) / 100.0
N0 = st.slider("Initial population N0", 1, 100, 10)
n_steps = st.slider("Number of steps", 0, 100, 20)

times, pops = simulate(N0, K, r, T, I, n_steps)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(times, pops, marker="o", color="tab:blue")
ax.set_xlabel("Time")
ax.set_ylabel("Population N")
ax.set_title("Logistic growth with periodic removal")
ax.grid(True)

st.pyplot(fig)
