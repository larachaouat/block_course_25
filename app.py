import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def logistic_curve(N0, K, r, T, n_points=1000):
    """Solve logistic curve for one interval of duration T."""
    t = np.linspace(0, T, n_points)
    N = K / (1 + (K / N0 - 1) * np.exp(-r * t))
    return t, N

def simulate(N0, K, r, T, I, n_steps, n_points=1000):
    times = []
    pops = []
    N = N0
    t_global = 0

    for step in range(1, n_steps + 1):
        # logistic growth within this interval
        t_local, N_local = logistic_curve(N, K, r, T, n_points)
        times.extend(t_global + t_local)
        pops.extend(N_local)

        # apply removal at the end
        N = (1 - I) * N_local[-1]
        t_global += T

        # store the removal jump as well
        times.append(t_global)
        pops.append(N)

    return np.array(times), np.array(pops)

# --- Streamlit interface ---
st.title("📈 Logistic Growth with Periodic Removal")

K = st.slider("Carrying capacity K", 10, 500, 100, 10)
r = st.slider("Growth rate r", 0.01, 1.0, 0.3, 0.01)
T = st.slider("Period T", 0.1, 5.0, 1.0, 0.1)
I = st.slider("Removal fraction I (%)", 0, 90, 20, 5) / 100.0
N0 = st.slider("Initial population N0", 1, 100, 10)
n_steps = st.slider("Number of steps", 1, 50, 10)

times, pops = simulate(N0, K, r, T, I, n_steps)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(times, pops, color="tab:blue")
ax.set_xlabel("Time")
ax.set_ylabel("Population N")
ax.set_title("Logistic growth with periodic removal")
ax.grid(True)

st.pyplot(fig)

