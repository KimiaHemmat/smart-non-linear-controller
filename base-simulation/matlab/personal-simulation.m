% personal-simulation.m
% Basic inverse pendulum simulation in MATLAB.

% System parameters
m = 1.0;          % Mass [kg]
l = 1.0;          % Pendulum length [m]
b = 0.08;         % Damping at the joint [N*m*s/rad]
g = 9.81;         % Gravity [m/s^2]
torque_limit = 2.0; % Actuator torque bound [N*m]

% Simulation settings
dt = 0.02;        % Integration step [s]
T = 12;           % Total simulation time [s]
N = floor(T/dt);  % Number of steps

% Controller settings: simple PD to upright
% Upright is set at theta = pi.
theta_ref = pi;
Kp = 30;
Kd = 6;

% Initial state
theta = zeros(1, N);
omega = zeros(1, N);
u = zeros(1, N);
time = (0:N-1) * dt;

theta(1) = pi + 0.3;   % 0.3 rad offset from upright
omega(1) = 0.0;

for k = 1:N-1
    % Error to upright and PD control.
    e = wrap_angle(theta(k) - theta_ref);
    u_cmd = -(Kp * e + Kd * omega(k));

    % Clip control torque.
    if u_cmd > torque_limit
        u(k) = torque_limit;
    elseif u_cmd < -torque_limit
        u(k) = -torque_limit;
    else
        u(k) = u_cmd;
    end

    % Add a small disturbance in the middle.
    d = 0;
    if k >= 300 && k <= 340
        d = 0.2;
    end

    % Inverse pendulum dynamics using Euler step.
    % theta_ddot = (u - b*omega - m*g*l*sin(theta)) / (m*l^2)
    alpha = (u(k) + d - b*omega(k) - m*g*l*sin(theta(k))) / (m*l^2);
    omega(k+1) = omega(k) + dt * alpha;
    theta(k+1) = theta(k) + dt * omega(k+1);
end

% Keep last control value for plot size match.
u(N) = u(N-1);

% Convert angle to [-pi, pi] before plotting
theta_error = arrayfun(@wrap_angle, theta - theta_ref);

% Plots
figure('Name', 'Inverse pendulum base simulation');
subplot(2,1,1);
plot(time, theta_error, 'LineWidth', 1.2);
grid on;
xlabel('Time [s]');
ylabel('Angle error [rad]');
title('Inverse pendulum angle error');

subplot(2,1,2);
plot(time, u, 'LineWidth', 1.2);
grid on;
xlabel('Time [s]');
ylabel('Control torque [N*m]');
title('Control input');

% wrap_angle: keep angle in [-pi, pi]
function a = wrap_angle(x)
    a = atan2(sin(x), cos(x));
end
