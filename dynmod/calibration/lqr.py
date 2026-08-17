"""Calibration tier (plan Part II 'Build: environments', last item).

Scalar and piecewise-affine linear systems with an analytic LQR expert whose
dependence on the hidden parameters c is written by hand. Purpose (plan
'Build: evaluation'): validate probes and ablations where the true mapping is
known. Both systems carry a dummy component in c that provably never enters
the dynamics or the expert - ablating a probe direction for it must produce
no effect, or the ablation is causing generic damage.

c vectors:
  ScalarSystem:          c = (mass, damping, dummy)
  PiecewiseAffineSystem: c = (mass, damping, stiffness, dummy)

Run `python -m dynmod.calibration.lqr` for a self-test: expert convergence
across the c range, gain sensitivity to each real component, gain invariance
to the dummy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def solve_dare_scalar(a: float, b: float, q: float, r: float) -> float:
    """Closed-form positive root of the scalar discrete algebraic Riccati
    equation P = a^2 P - (a b P)^2 / (r + b^2 P) + q, i.e. of
    b^2 P^2 + (r - a^2 r - q b^2) P - q r = 0."""
    A = b * b
    B = r - a * a * r - q * b * b
    C = -q * r
    return (-B + np.sqrt(B * B - 4 * A * C)) / (2 * A)


def solve_dare(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
    iters: int = 10000, tol: float = 1e-14,
) -> np.ndarray:
    """Fixed-point iteration of the DARE to machine precision (exact solution;
    the hand-written part is A(c), B(c), through which c enters)."""
    P = Q.copy()
    for _ in range(iters):
        BT_P = B.T @ P
        K = np.linalg.solve(R + BT_P @ B, BT_P @ A)
        P_new = Q + A.T @ P @ (A - B @ K)
        if np.max(np.abs(P_new - P)) < tol:
            return P_new
        P = P_new
    return P


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = solve_dare(A, B, Q, R)
    BT_P = B.T @ P
    return np.linalg.solve(R + BT_P @ B, BT_P @ A)


@dataclass
class ScalarSystem:
    """x_{t+1} = a(c) x_t + b(c) u_t with a = 1 - dt*damping/mass, b = dt/mass.

    c = (mass, damping, dummy). The dummy never appears in a, b, or the gain.
    """

    dt: float = 0.05
    q: float = 1.0
    r: float = 0.1
    noise_std: float = 0.0
    mass_range: tuple = (0.5, 2.0)
    damping_range: tuple = (0.1, 1.0)

    c_dim = 3
    x_dim = 1
    u_dim = 1

    def sample_c(self, n: int, rng: np.random.Generator) -> np.ndarray:
        mass = np.exp(rng.uniform(*np.log(self.mass_range), size=n))
        damping = np.exp(rng.uniform(*np.log(self.damping_range), size=n))
        dummy = rng.uniform(-1.0, 1.0, size=n)
        return np.stack([mass, damping, dummy], axis=1)

    # hand-written dependence of the dynamics on c
    def a(self, c: np.ndarray) -> np.ndarray:
        return 1.0 - self.dt * c[..., 1] / c[..., 0]

    def b(self, c: np.ndarray) -> np.ndarray:
        return self.dt / c[..., 0]

    def step(self, x: np.ndarray, u: np.ndarray, c: np.ndarray,
             rng: np.random.Generator | None = None) -> np.ndarray:
        x_new = self.a(c) * x + self.b(c) * u
        if self.noise_std > 0 and rng is not None:
            x_new = x_new + rng.normal(0, self.noise_std, size=x.shape)
        return x_new

    # hand-written dependence of the expert on c (closed-form Riccati root)
    def expert_gain(self, c: np.ndarray) -> np.ndarray:
        a, b = self.a(c), self.b(c)
        a, b = np.atleast_1d(a), np.atleast_1d(b)
        K = np.empty_like(a)
        for i in range(a.shape[0]):
            P = solve_dare_scalar(a[i], b[i], self.q, self.r)
            K[i] = a[i] * b[i] * P / (self.r + b[i] * b[i] * P)
        return K

    def expert_action(self, x: np.ndarray, c: np.ndarray, x_goal: float = 0.0) -> np.ndarray:
        return -self.expert_gain(c) * (x - x_goal)

    def rollout(self, x0: np.ndarray, c: np.ndarray, T: int,
                rng: np.random.Generator | None = None, x_goal: float = 0.0):
        """Batched expert rollout. Returns (states [T+1, n], actions [T, n])."""
        n = x0.shape[0]
        xs = np.empty((T + 1, n))
        us = np.empty((T, n))
        xs[0] = x0
        K = self.expert_gain(c)
        for t in range(T):
            us[t] = -K * (xs[t] - x_goal)
            us[t] = np.asarray(us[t])
            xs[t + 1] = self.step(xs[t], us[t], c, rng)
        return xs, us


@dataclass
class PiecewiseAffineSystem:
    """Point mass on a line with a one-sided spring wall at p = wall.

    state x = (p, v); control u = force.
      free   (p <  wall): v' = v + dt*(u - damping*v) / mass
      contact(p >= wall): v' = v + dt*(u - damping*v - stiffness*(p - wall)) / mass
    p' = p + dt*v in both modes: two affine modes A_free(c), A_contact(c).

    c = (mass, damping, stiffness, dummy). The goal sits inside the contact
    region, so holding it requires the feedforward force u_ff = stiffness *
    (p_goal - wall): the expert's c-dependence is explicit in both the
    per-mode LQR gains and the feedforward term.
    """

    dt: float = 0.02
    wall: float = 1.0
    p_goal: float = 1.05
    q_pos: float = 10.0
    q_vel: float = 1.0
    r: float = 0.05
    noise_std: float = 0.0
    mass_range: tuple = (0.5, 2.0)
    damping_range: tuple = (0.1, 1.0)
    stiffness_range: tuple = (20.0, 80.0)

    c_dim = 4
    x_dim = 2
    u_dim = 1

    def sample_c(self, n: int, rng: np.random.Generator) -> np.ndarray:
        mass = np.exp(rng.uniform(*np.log(self.mass_range), size=n))
        damping = np.exp(rng.uniform(*np.log(self.damping_range), size=n))
        stiff = np.exp(rng.uniform(*np.log(self.stiffness_range), size=n))
        dummy = rng.uniform(-1.0, 1.0, size=n)
        return np.stack([mass, damping, stiff, dummy], axis=1)

    # hand-written mode dynamics
    def _AB(self, c_row: np.ndarray, contact: bool):
        m, d, k = c_row[0], c_row[1], c_row[2]
        dt = self.dt
        A = np.array(
            [[1.0, dt],
             [(-dt * k / m if contact else 0.0), 1.0 - dt * d / m]]
        )
        B = np.array([[0.0], [dt / m]])
        return A, B

    def step(self, x: np.ndarray, u: np.ndarray, c: np.ndarray,
             rng: np.random.Generator | None = None) -> np.ndarray:
        """x: (n, 2), u: (n,), c: (n, 4)."""
        p, v = x[:, 0], x[:, 1]
        m, d, k = c[:, 0], c[:, 1], c[:, 2]
        contact = p >= self.wall
        spring = np.where(contact, -k * (p - self.wall), 0.0)
        v_new = v + self.dt * (u - d * v + spring) / m
        p_new = p + self.dt * v
        out = np.stack([p_new, v_new], axis=1)
        if self.noise_std > 0 and rng is not None:
            out = out + rng.normal(0, self.noise_std, size=out.shape)
        return out

    def expert_gains(self, c_row: np.ndarray):
        """Per-mode LQR gains for one c. Depends only on (mass, damping,
        stiffness); the dummy component provably never enters."""
        Q = np.diag([self.q_pos, self.q_vel])
        R = np.array([[self.r]])
        A_f, B_f = self._AB(c_row, contact=False)
        A_c, B_c = self._AB(c_row, contact=True)
        return lqr_gain(A_f, B_f, Q, R), lqr_gain(A_c, B_c, Q, R)

    def batch_gains(self, c: np.ndarray):
        """Precompute per-env (K_free, K_contact) once; the gains depend only
        on c, so recomputing them per step would be pure waste."""
        return [self.expert_gains(c[i]) for i in range(c.shape[0])]

    def expert_action(self, x: np.ndarray, c: np.ndarray, gains=None) -> np.ndarray:
        """Batched piecewise LQR with hand-written feedforward hold force."""
        n = x.shape[0]
        if gains is None:
            gains = self.batch_gains(c)
        u = np.empty(n)
        x_goal = np.array([self.p_goal, 0.0])
        for i in range(n):
            K_free, K_contact = gains[i]
            in_contact = x[i, 0] >= self.wall
            K = K_contact if in_contact else K_free
            u_ff = c[i, 2] * (self.p_goal - self.wall) if in_contact else 0.0
            u[i] = u_ff - (K @ (x[i] - x_goal)).item()
        return u

    def rollout(self, x0: np.ndarray, c: np.ndarray, T: int,
                rng: np.random.Generator | None = None):
        n = x0.shape[0]
        gains = self.batch_gains(c)
        xs = np.empty((T + 1, n, 2))
        us = np.empty((T, n))
        xs[0] = x0
        for t in range(T):
            us[t] = self.expert_action(xs[t], c, gains=gains)
            xs[t + 1] = self.step(xs[t], us[t], c, rng)
        return xs, us


def _self_test():
    rng = np.random.default_rng(0)
    print("== ScalarSystem ==")
    sys1 = ScalarSystem()
    c = sys1.sample_c(64, rng)
    x0 = rng.uniform(-2, 2, size=64)
    xs, _ = sys1.rollout(x0, c, T=120)
    final_err = np.abs(xs[-1]).max()
    print(f"expert final |x| max over 64 c draws: {final_err:.2e}")
    assert final_err < 1e-3, "scalar expert failed to converge"
    K = sys1.expert_gain(c)
    c_dummy = c.copy()
    c_dummy[:, 2] = rng.uniform(-1, 1, size=64)
    assert np.allclose(K, sys1.expert_gain(c_dummy)), "gain depends on dummy!"
    c_hi = c.copy(); c_hi[:, 0] *= 2.0
    assert not np.allclose(K, sys1.expert_gain(c_hi)), "gain ignores mass!"
    print("gain varies with mass, invariant to dummy: OK")

    print("== PiecewiseAffineSystem ==")
    sys2 = PiecewiseAffineSystem()
    c = sys2.sample_c(32, rng)
    x0 = np.stack([rng.uniform(-0.5, 0.5, 32), np.zeros(32)], axis=1)
    xs, _ = sys2.rollout(x0, c, T=600)
    final_err = np.abs(xs[-1, :, 0] - sys2.p_goal).max()
    final_vel = np.abs(xs[-1, :, 1]).max()
    print(f"expert final |p - p_goal| max: {final_err:.2e}, |v| max: {final_vel:.2e}")
    assert final_err < 5e-3, "piecewise expert failed to reach the in-contact goal"
    K_f0, K_c0 = sys2.expert_gains(c[0])
    c_alt = c[0].copy(); c_alt[3] = -c_alt[3] + 0.1
    K_f1, K_c1 = sys2.expert_gains(c_alt)
    assert np.allclose(K_f0, K_f1) and np.allclose(K_c0, K_c1), "gain depends on dummy!"
    c_stiff = c[0].copy(); c_stiff[2] *= 2.0
    assert not np.allclose(K_c0, sys2.expert_gains(c_stiff)[1]), "contact gain ignores stiffness!"
    assert np.allclose(K_f0, sys2.expert_gains(c_stiff)[0]), "free gain should ignore stiffness!"
    print("contact gain varies with stiffness, free gain does not, both invariant to dummy: OK")
    print("calibration tier self-test PASSED")


if __name__ == "__main__":
    _self_test()
