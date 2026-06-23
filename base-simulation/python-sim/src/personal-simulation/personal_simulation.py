"""
personal_simulation.py
Simple inverse pendulum simulation in Python.
This version shows an interactive animation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def wrap_angle(x):
    """
    Keep angle in [-pi, pi].
    """
    return np.arctan2(np.sin(x), np.cos(x))


def run_simulation():
    # Parameters
    m = 1.0
    l = 1.0
    b = 0.08
    g = 9.81
    torque_limit = 2.0

    # Time settings
    dt = 0.02
    T = 12.0
    N = int(T / dt)

    # Controller settings: simple PD to upright
    theta_ref = np.pi
    kp = 30.0
    kd = 6.0

    # Storage
    theta = np.zeros(N)
    omega = np.zeros(N)
    u = np.zeros(N)
    t = np.arange(N) * dt

    # Initial state
    theta[0] = np.pi + 0.3
    omega[0] = 0.0

    for k in range(N - 1):
        # PD control
        err = wrap_angle(theta[k] - theta_ref)
        u_cmd = -(kp * err + kd * omega[k])

        # Clip torque
        if u_cmd > torque_limit:
            u[k] = torque_limit
        elif u_cmd < -torque_limit:
            u[k] = -torque_limit
        else:
            u[k] = u_cmd

        # Small disturbance in the middle
        d = 0.2 if 300 <= k <= 340 else 0.0

        # Inverse pendulum dynamics with Euler integration
        alpha = (u[k] + d - b * omega[k] - m * g * l * np.sin(theta[k])) / (m * l**2)
        omega[k + 1] = omega[k] + dt * alpha
        theta[k + 1] = theta[k] + dt * omega[k + 1]

    u[-1] = u[-2]
    theta_err = wrap_angle(theta - theta_ref)

    return t, theta, omega, u, theta_err, l, dt, torque_limit


def main():
    t, theta, omega, u, theta_err, l, dt, torque_limit = run_simulation()

    # Make interactive plots
    fig = plt.figure(figsize=(9, 8))

    # Top: animated pendulum
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title("Inverse pendulum animation")
    ax1.set_xlim(-1.2 * l, 1.2 * l)
    ax1.set_ylim(-1.2 * l, 1.2 * l)
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True)

    rod, = ax1.plot([], [], "o-", lw=3)
    trace_x = []
    trace_y = []
    trace, = ax1.plot([], [], "r-", lw=1)

    # Bottom: error and control while running
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_title("Tracking error and control")
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(-max(1.0, np.max(np.abs(theta_err)) + 0.3), max(1.0, np.max(np.abs(theta_err)) + 0.3))
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Rad / Nm (scaled view)")
    ax2.grid(True)

    err_line, = ax2.plot([], [], label="angle error [rad]", lw=1.2)
    # Separate y-axis keeps this simple and readable
    ax2_u = ax2.twinx()
    ax2_u.set_ylim(-1.2 * torque_limit, 1.2 * torque_limit)
    ax2_u.set_ylabel("Control torque [N*m]")
    u_line, = ax2_u.plot([], [], "r", alpha=0.7, label="control")

    time_text = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, va="top")
    leg_items = [err_line, u_line]
    ax2.legend(handles=leg_items, loc="upper right")

    # Start with the first point
    x0 = [0, l * np.sin(theta[0])]
    y0 = [0, -l * np.cos(theta[0])]
    rod.set_data(x0, y0)
    trace_x.append(x0[1])
    trace_y.append(y0[1])
    trace.set_data(trace_x, trace_y)

    def init():
        rod.set_data([], [])
        trace.set_data([], [])
        err_line.set_data([], [])
        u_line.set_data([], [])
        time_text.set_text("")
        return rod, trace, err_line, u_line, time_text

    def update(frame):
        x = [0, l * np.sin(theta[frame])]
        y = [0, -l * np.cos(theta[frame])]
        rod.set_data(x, y)

        trace_x.append(x[1])
        trace_y.append(y[1])
        trace.set_data(trace_x, trace_y)

        err_line.set_data(t[:frame + 1], theta_err[:frame + 1])
        u_line.set_data(t[:frame + 1], u[:frame + 1])
        time_text.set_text(f"t={t[frame]:.2f}s, err={theta_err[frame]:.3f}, u={u[frame]:.2f}")
        return rod, trace, err_line, u_line, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=dt * 1000,
        blit=False,
        repeat=False,
    )

    # Keep animation from being garbage collected
    plt.tight_layout()
    plt.show()
    return anim


if __name__ == "__main__":
    main()
