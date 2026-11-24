"""Relay cascade (N-i switching) control of a DC positioning drive.

Plant data, limits and controller structure follow Sadovoi A. V., Sukhinin B. V., Sokhina Yu. V.,
Derets A. L. "Relay systems of optimal control of electric drives" (DSTU, 2011), Ch. 6-7.
Equation numbers in comments refer to the paper (Eqs. (1)-(8)).

Output-shaft model, Eq. (1):
    dtheta/dt = w
    dw/dt     = b * (M - Mc),                 b = 0.182 / kappa   (kappa = J / J_nom)
    dM/dt     = a32 * w + a33 * M + m3 * U,   |U| <= Umax
Error and its derivatives, Eq. (2): E1 = theta* - theta, E2 = -w, E3 = -b (M - Mc).
Relay cascade, Eq. (3), N = 3:
    s1 = E1 + K12 E2 + K13 E3          E2* = -V sign(s1)
    s2 = E2 - E2* + K23 E3             E3* = -A sign(s2)
    s3 = E3 - E3*                      U   = Umax sign(s3)
Run this file to check the analytic synthesis against the published coefficients (Table 1).
"""
import numpy as np
from numba import njit

# --- nominal drive data at the output shaft (monograph, Table 7.1) ---
B23 = 0.182        # 1/J_nom
A32 = -16435.0     # back-EMF feedback
A33 = -126.1       # -R/L
M3 = 845.0         # input gain
UMAX = 1.1 * 27.0  # voltage limit (forcing factor 1.1)
WMAX = 1.19        # admissible speed, rad/s
MMAX = 25.2        # admissible torque, N*m
TOLF = 0.01        # settling band: |E1| <= 1 % of the displacement, Eq. (8)


def design_limits(kappa_hat=1.0):
    """Design limits (E2max, E3max, E4max) of Eq. (4) for an assumed inertia ratio kappa_hat."""
    b = B23 / kappa_hat
    V = WMAX
    A = b * MMAX                 # 4.586 rad/s^2 at kappa_hat = 1
    J = b * M3 * UMAX            # 4568 rad/s^3 at kappa_hat = 1 (the monograph rounds to 4.6e3)
    return V, A, J


def ni_coefficients(V, A, J):
    """Analytic N-i switching coefficients (K12, K13, K23), Eqs. (5)-(6):
    characteristic points of the jerk-limited time-optimal design trajectory."""
    K23 = A / (2.0 * J)
    K12 = V / (2.0 * A) + A / (2.0 * J)
    vb = A * A / (2.0 * J)
    db = A ** 3 / (6.0 * J * J)
    K13 = (K12 * vb - db) / A
    return np.array([K12, K13, K23])


def t_ref(dist, V, A, J):
    """Kinematic reference T*: minimum rest-to-rest time of a triple integrator
    with |v| <= V, |a| <= A, |j| <= J (jerk-limited S-curve)."""
    def half(vp):
        ta = vp / A + A / J if vp >= A * A / J else 2.0 * np.sqrt(vp / J)
        return ta, vp * ta / 2.0   # duration and distance of the acceleration half
    ta, da = half(V)
    if dist >= 2 * da:
        return 2 * ta + (dist - 2 * da) / V
    lo, hi = 0.0, V
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if 2 * half(mid)[1] < dist:
            lo = mid
        else:
            hi = mid
    return 2 * half(0.5 * (lo + hi))[0]


def t_ref_band(dist, V, A, J, band=0.01, dt=2e-6):
    """Kinematic reference T*_1%: time at which the S-curve of t_ref enters the band |A - x| <= band * A."""
    T = t_ref(dist, V, A, J)
    lo, hi = 0.0, V

    def dist_of(vp):
        ta = vp / A + A / J if vp >= A * A / J else 2.0 * np.sqrt(vp / J)
        return vp * ta
    if dist >= dist_of(V):
        vp = V
    else:
        for _ in range(100):
            m = 0.5 * (lo + hi)
            if dist_of(m) < dist:
                lo = m
            else:
                hi = m
        vp = 0.5 * (lo + hi)
    ap = min(A, np.sqrt(vp * J))
    tj = ap / J
    ta = vp / ap - ap / J if vp >= A * A / J else 0.0
    # integrate the deceleration half backwards from the final rest state (symmetric profile)
    t = 0.0; a = 0.0; v = 0.0; x = 0.0
    target = band * dist
    for seg, jerk in ((tj, J), (max(ta, 0.0), 0.0), (tj, -J)):
        n = int(np.ceil(seg / dt))
        h = seg / n if n else 0.0
        for _ in range(n):
            a += jerk * h; v += a * h; x += v * h; t += h
            if x >= target:
                return T - t
    return T - t


@njit(cache=True)
def simulate(theta_ref, K12, K13, K23, V, A, Jd, kappa, Mc, T, dt, rec_every, tolf):
    """Explicit Euler simulation of the drive (1) with the relay cascade (3) (compiled by Numba)."""
    b = B23 / kappa
    n = int(T / dt)
    th = 0.0; w = 0.0; M = 0.0
    nrec = n // rec_every + 1
    rec = np.zeros((nrec, 6))
    sw1 = 0; sw2 = 0
    last1 = 0.0; last2 = 0.0
    Mpeak = 0.0; ovs = 0.0; wpk = 0.0
    ts = 0.0
    tol = max(tolf * theta_ref, 1e-6)
    k = 0
    for i in range(n):
        E1 = theta_ref - th
        E2 = -w
        E3 = -b * (M - Mc)
        s1 = E1 + K12 * E2 + K13 * E3
        u1 = 1.0 if s1 > 0 else -1.0
        E2s = -V * u1
        s2 = E2 - E2s + K23 * E3
        u2 = 1.0 if s2 > 0 else -1.0
        E3s = -A * u2
        s3 = E3 - E3s
        U = UMAX if s3 > 0 else -UMAX
        if i > 0:
            if u1 != last1: sw1 += 1
            if u2 != last2: sw2 += 1
        last1 = u1; last2 = u2
        dth = w
        dw = b * (M - Mc)
        dM = A32 * w + A33 * M + M3 * U
        th += dt * dth; w += dt * dw; M += dt * dM
        if abs(M) > Mpeak: Mpeak = abs(M)
        if abs(w) > wpk: wpk = abs(w)
        if th - theta_ref > ovs: ovs = th - theta_ref
        if abs(theta_ref - th) > tol: ts = (i + 1) * dt     # last time outside the band, Eq. (8)
        if i % rec_every == 0:
            rec[k, 0] = i * dt; rec[k, 1] = th; rec[k, 2] = w; rec[k, 3] = M
            rec[k, 4] = U; rec[k, 5] = s1
            k += 1
    return ts, ovs, Mpeak, sw1, sw2, rec[:k], wpk


def run(theta_ref, K, kappa=1.0, kappa_hat=None, Mc=0.0, dt=1e-5, rec_every=100, T=None, tolf=None):
    """Simulate one rest-to-rest move of amplitude theta_ref [rad] with K = (K12, K13, K23).

    kappa      true inertia ratio J / J_nom of the simulated drive;
    kappa_hat  inertia ratio assumed by the controller (sets the relay amplitudes, Eq. (4)).
    Returns a dict: ts (settling time, inf if not settled), ovs (overshoot, fraction of theta_ref),
    Mpeak [N*m], wpk [rad/s], sw1/sw2 (numbers of relay switchings), Tref (T* of the assumed
    limits) and rec = [t, theta, omega, M, U, s1] sampled every rec_every steps.
    """
    kh = kappa if kappa_hat is None else kappa_hat
    V, A, J = design_limits(kh)
    if T is None:
        T = 1.6 * t_ref(theta_ref, V, A, J) + 0.4
    ts, ovs, Mp, sw1, sw2, rec, wpk = simulate(theta_ref, K[0], K[1], K[2], V, A, J,
                                               kappa, Mc, T, dt, rec_every, TOLF if tolf is None else tolf)
    settled = ts < T - 0.05
    return dict(ts=ts if settled else np.inf, ovs=ovs / theta_ref, Mpeak=Mp, wpk=wpk,
                sw1=sw1, sw2=sw2, Tref=t_ref(theta_ref, V, A, J), rec=rec)


if __name__ == "__main__":
    for label, lim in (("Triple-integrator example (limits 2, 3, 10)", (2, 3, 10)),
                       ("Monograph drive, limits (1.19, 4.58, 4.6e3) -> matrix (7.4)", (1.19, 4.58, 4600.0))):
        K = ni_coefficients(*lim)
        print(f"{label}: K = {K}; roots of K13 p^2 + K12 p + 1 = 0: {np.roots([K[1], K[0], 1.0]).real.round(2)}; "
              f"-1/K23 = {-1 / K[2]:.2f}")
    print("Limits (4) at kappa_hat = 1:", design_limits(1.0), "-> K =", ni_coefficients(*design_limits(1.0)))
    K0 = ni_coefficients(*design_limits(1.0))
    for th in [3.1416, 1.0, 0.3, 0.1, 0.03, 0.01]:
        r = run(th, K0)
        print(f"A={th:7.4f} ts={r['ts']:.4f} T*={r['Tref']:.4f} tau={r['ts'] / r['Tref']:.3f} "
              f"overshoot={100 * r['ovs']:.3f}% Mpk={r['Mpeak']:.1f} sw1={r['sw1']} sw2={r['sw2']}")
