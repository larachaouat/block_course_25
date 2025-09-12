import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def logistic_curve(Nprev, K, r,  T, n_points):
    """Solve logistic curve for one interval of duration T."""
    t = np.linspace(0, T, n_points)
    N = K / (1 + (K / Nprev - 1) * np.exp(-r * t))
    return t, N

def simulate(N0, K, r, T, I, T_end=100, n_points=100):
    times = []
    pops = []
    N_prev = N0
    t_global = 0
    n_steps = int(T_end/T)

    for step in range(1, n_steps + 2):
        if t_global+T> T_end:
            T= T_end-(step-1)*T
            print(T)
        # logistic growth within this interval
        t_local, N_local = logistic_curve(N_prev, K, r, T, n_points)
        times.extend(t_global + t_local)
        pops.extend(N_local)

        # apply removal at the end
        N_prev = (1 - I) * N_local[-1]
        t_global += T

        # store the removal jump as well
        if t_global+T<= T_end:
            times.append(t_global)
            pops.append(N_prev)

    return np.array(times), np.array(pops)
    
def simulate_variable_removal(N0, K, r, T, I_mean, I_var, T_end=100, n_points=100):
    """Simulate logistic growth with periodic removal, variable intensity."""
    times = []
    pops = []
    N_prev = N0
    t_global = 0
    n_steps = int(T_end/T)


    for step in range(1, n_steps + 2):
        if t_global + T > T_end:
            T = T_end - (step-1) * T

        t_local, N_local = logistic_curve(N_prev, K, r, T, n_points)
        times.extend(t_global + t_local)
        pops.extend(N_local)

        # sample removal fraction from normal distribution
        I_actual =I_mean + I_var *(-1)**step 
        print(I_actual,n_steps)
        N_prev = (1 - I_actual) * N_local[-1]
        t_global += T

        # store the removal jump
        if t_global + T <= T_end:
            times.append(t_global)
            pops.append(N_prev)

    return np.array(times), np.array(pops)


# --- Streamlit interface ---
st.title("📈 Logistic Growth with Periodic Removal")

K = st.slider("Carrying capacity K", 1, 100, 10, 1)
r = st.slider("Growth rate r", 0.0, 5.0, 0.3, 0.01)
T = st.slider("Period T", 0.1, 100.0, 1.0, 1.0)
I = st.slider("Removal fraction I (%)", 0, 100, 20, 5) / 100.0
N0 = st.slider("Initial population N0", 1, 100, 10)
T_end = st.slider("End simulation", 1, 1000, 100, 10)
# New parameter for variable removal
I_var = st.slider("Removal variability (%)", 0, I*100, 5, 1) / 100.0

# --- Simulations ---
# deterministic removal
times_det, pops_det = simulate(N0, K, r, T, I, T_end)

# variable removal
times_var, pops_var = simulate_variable_removal(N0, K, r, T, I, I_var, T_end)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(times_det, pops_det, label="Fixed removal", color="tab:blue")
ax.plot(times_var, pops_var, label="Variable removal", color="tab:red", alpha=0.7)
ax.set_xlabel("Time")
ax.set_ylabel("Population N")
ax.set_title("Logistic growth with periodic removal")
ax.grid(True)
ax.legend()

st.pyplot(fig)
