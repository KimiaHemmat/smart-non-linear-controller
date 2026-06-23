"""
official_simulation.py
Official Gymnasium inverted pendulum demo with a cleaner UI.
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button, Slider


def wrap_angle(x):
    return np.arctan2(np.sin(x), np.cos(x))


def pd_action(state, kp=4.0, kd=0.7):
    """
    Small PD controller for Pendulum-v1 state [cos(theta), sin(theta), theta_dot].
    """
    ct, st, w = state
    theta = np.arctan2(st, ct)
    err = wrap_angle(theta)  # target theta = 0 (upright in gym env)
    u = -(kp * err + kd * w)
    u = np.clip(u, -2.0, 2.0)
    return np.array([u], dtype=np.float32)


def simulate_data(steps=900, seed=2):
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=seed)
    dt = 1.0 / 50.0

    t = np.zeros(steps)
    theta = np.zeros(steps)
    omega = np.zeros(steps)
    u_hist = np.zeros(steps)
    reward_hist = np.zeros(steps)

    total_reward = 0.0
    n = steps

    for k in range(steps):
        action = pd_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        ct, st, w = obs
        theta[k] = np.arctan2(st, ct)
        omega[k] = w
        u_hist[k] = float(action[0])
        reward_hist[k] = float(reward)
        t[k] = k * dt
        total_reward += reward_hist[k]

        if terminated or truncated:
            n = k + 1
            t = t[:n]
            theta = theta[:n]
            omega = omega[:n]
            u_hist = u_hist[:n]
            reward_hist = reward_hist[:n]
            break

    env.close()
    print(f"Episode reward: {total_reward:.2f}")
    return t, theta, omega, u_hist, reward_hist


def main():
    t, theta, omega, u_hist, reward_hist = simulate_data()
    l = 1.0
    n = len(t)

    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(left=0.24, right=0.98, top=0.96, bottom=0.12, hspace=0.35, wspace=0.28)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])

    ax_anim = fig.add_subplot(gs[0, :])
    ax_signals = fig.add_subplot(gs[1, 0])
    ax_control = fig.add_subplot(gs[1, 1])

    # Animation view
    ax_anim.set_title("Pendulum animation (Gymnasium Pendulum-v1)")
    ax_anim.set_xlim(-1.2 * l, 1.2 * l)
    ax_anim.set_ylim(-1.2 * l, 1.2 * l)
    ax_anim.set_aspect("equal", adjustable="box")
    ax_anim.grid(True)
    rod, = ax_anim.plot([], [], "o-", lw=3)
    trail_x, trail_y = [], []
    trail, = ax_anim.plot([], [], "k-", lw=1)

    # Signal plots
    ax_signals.set_title("Angle and velocity")
    ax_signals.set_xlabel("time [s]")
    ax_signals.set_ylabel("angle [rad]")
    ax_signals.grid(True)
    ax_signals.set_xlim(t[0], t[-1] if n > 1 else 1.0)
    ymax_angle = max(2.5, np.max(np.abs(theta)) + 0.3)
    ax_signals.set_ylim(-ymax_angle, ymax_angle)
    ln_theta, = ax_signals.plot([], [], label="theta", lw=1.3)

    ax_vel = ax_signals.twinx()
    ymax_vel = max(4.0, np.max(np.abs(omega)) + 1.0)
    ax_vel.set_ylim(-ymax_vel, ymax_vel)
    ax_vel.set_ylabel("omega [rad/s]")
    ln_omega, = ax_vel.plot([], [], "tab:orange", alpha=0.75, label="omega", lw=1.1)

    # Control plot
    ax_control.set_title("Control torque")
    ax_control.set_xlabel("time [s]")
    ax_control.set_ylabel("u [N*m]")
    ax_control.set_xlim(t[0], t[-1] if n > 1 else 1.0)
    ax_control.set_ylim(-2.2, 2.2)
    ax_control.grid(True)
    ln_u, = ax_control.plot([], [], label="u", lw=1.3)

    # Legend
    ax_signals.legend(handles=[ln_theta, ln_omega], loc="upper right")
    ax_control.legend(loc="upper right")

    # Text status area
    status = fig.text(0.34, 0.01, "", fontsize=9)

    state = {"i": 0, "playing": False, "step": 3}

    def reset_scene():
        state["i"] = 0
        state["playing"] = False
        trail_x.clear()
        trail_y.clear()
        rod.set_data([], [])
        trail.set_data([], [])
        ln_theta.set_data([], [])
        ln_omega.set_data([], [])
        ln_u.set_data([], [])

        x0 = [0, l * np.sin(theta[0])]
        y0 = [0, -l * np.cos(theta[0])]
        rod.set_data(x0, y0)
        trail_x.append(x0[1])
        trail_y.append(y0[1])
        trail.set_data(trail_x, trail_y)
        fig.canvas.draw_idle()

    def draw_frame(i):
        x = [0, l * np.sin(theta[i])]
        y = [0, -l * np.cos(theta[i])]
        rod.set_data(x, y)

        trail_x.append(x[1])
        trail_y.append(y[1])
        trail.set_data(trail_x, trail_y)

        ln_theta.set_data(t[: i + 1], theta[: i + 1])
        ln_omega.set_data(t[: i + 1], omega[: i + 1])
        ln_u.set_data(t[: i + 1], u_hist[: i + 1])

        if i < len(reward_hist):
            total_so_far = float(np.sum(reward_hist[: i + 1]))
        else:
            total_so_far = float(np.sum(reward_hist))

        status.set_text(
            f"step={i+1}/{n}   t={t[i]:.2f}s   "
            f"theta={theta[i]:.2f}   omega={omega[i]:.2f}   "
            f"u={u_hist[i]:.2f}   reward={total_so_far:.1f}"
        )

    def update(_):
        if not state["playing"]:
            return rod, trail, ln_theta, ln_omega, ln_u

        i = state["i"]
        if i >= n:
            state["playing"] = False
            status.set_text("finished - press Reset to replay")
            return rod, trail, ln_theta, ln_omega, ln_u

        draw_frame(i)
        state["i"] = i + state["step"]
        return rod, trail, ln_theta, ln_omega, ln_u

    # Control box (top-left corner)
    ax_box = fig.add_axes([0.03, 0.68, 0.18, 0.28])
    ax_box.set_xticks([])
    ax_box.set_yticks([])
    ax_box.set_xlim(0, 1)
    ax_box.set_ylim(0, 1)
    ax_box.set_title("Controls", fontsize=10, pad=8)
    ax_box.set_frame_on(True)
    ax_box.patch.set_facecolor("#f5f5f5")
    for side in ax_box.spines.values():
        side.set_visible(True)

    # Put buttons inside the box
    ax_play = fig.add_axes([0.05, 0.82, 0.13, 0.05], facecolor="lightgoldenrodyellow")
    ax_pause = fig.add_axes([0.05, 0.74, 0.13, 0.05], facecolor="lightgoldenrodyellow")
    ax_reset = fig.add_axes([0.05, 0.66, 0.13, 0.05], facecolor="lightgoldenrodyellow")
    ax_speed = fig.add_axes([0.05, 0.56, 0.12, 0.05])

    btn_play = Button(ax_play, "Play")
    btn_pause = Button(ax_pause, "Pause")
    btn_reset = Button(ax_reset, "Reset")
    speed = Slider(ax_speed, "Speed", 1, 8, valinit=3, valstep=1)
    speed.valtext.set_text("x3")

    def on_play(_):
        if state["i"] >= n:
            reset_scene()
        state["playing"] = True

    def on_pause(_):
        state["playing"] = False

    def on_reset(_):
        state["playing"] = False
        reset_scene()

    def on_speed_change(val):
        state["step"] = int(val)

    btn_play.on_clicked(on_play)
    btn_pause.on_clicked(on_pause)
    btn_reset.on_clicked(on_reset)
    speed.on_changed(on_speed_change)

    # Start paused with first frame visible
    state["i"] = 0
    reset_scene()
    status.set_text("ready")

    anim = animation.FuncAnimation(
        fig,
        update,
        interval=20,
        blit=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
