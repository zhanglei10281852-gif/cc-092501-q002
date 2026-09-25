"""端元混合反演的加权最小二乘求解器（纯标准库实现）。

模型：对 m 个有效示踪指标、n 个补给端元，
    r(f) = A f - y,    f >= 0,  sum(f) = 1
端元取值本身带误差时，混合预测值的协方差随比例变化：
    C(f) = C_obs + sum_j f_j^2 C_j
目标（卡方）：
    phi(f) = r^T C(f)^-1 r
梯度：
    d phi / d f_j = 2 a_j^T u - 2 f_j u^T C_j u,   u = C^-1 r
使用带 Armijo 回溯的投影梯度法（投影到标准单纯形），过程确定、无随机量。
"""

from __future__ import annotations

import math
from typing import Any

TRACERS = ("isotope_d18o", "isotope_d2h", "solute_mg_l")
SOLVER_VERSION = "wls-mix-2"

# 有效秩阈值：lambda < lambda_max * RANK_EPS 视为零方向
RANK_EPS = 1e-10
# 条件数超过该值视为数值不可辨识
ILL_CONDITIONED = 1e12
# 两正交流端元在加权示踪空间中距离平方小于该值视为近似重合
COLLINEAR_DIST2 = 1e-8
OUTLIER_Z = 3.0


class CovarianceError(ValueError):
    """协方差矩阵形状、对称性或正定性不满足要求。"""


# ---------------------------------------------------------------- 基础线性代数

def zeros(rows: int, cols: int | None = None) -> list[list[float]]:
    cols = rows if cols is None else cols
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def cholesky(a: list[list[float]]) -> list[list[float]]:
    """返回下三角 L 使 L L^T = a；a 必须正定。"""
    n = len(a)
    lower = zeros(n)
    for i in range(n):
        for j in range(i + 1):
            total = a[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if total <= 1e-14:
                    raise CovarianceError("covariance_not_positive_definite")
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    return lower


def cholesky_solve(lower: list[list[float]], rhs: list[float]) -> list[float]:
    """解 L L^T x = rhs。"""
    n = len(lower)
    y = [0.0] * n
    for i in range(n):
        y[i] = (rhs[i] - sum(lower[i][k] * y[k] for k in range(i))) / lower[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(lower[k][i] * x[k] for k in range(i + 1, n))) / lower[i][i]
    return x


def solve_spd(a: list[list[float]], rhs: list[float]) -> list[float]:
    return cholesky_solve(cholesky(a), rhs)


def quadratic(a: list[list[float]], x: list[float], y: list[float] | None = None) -> float:
    y = x if y is None else y
    return sum(x[i] * sum(a[i][k] * y[k] for k in range(len(a))) for i in range(len(a)))


def validate_covariance3(matrix: list[list[float]], context: str) -> None:
    """校验 3x3 协方差：形状、有限值、对称、对角非负，有效子块正定。"""
    label = f"{context}_covariance"
    if not isinstance(matrix, list) or len(matrix) != 3 or any(
        not isinstance(row, list) or len(row) != 3 for row in matrix
    ):
        raise CovarianceError(f"{label}_shape_must_be_3x3")
    flat = [value for row in matrix for value in row]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in flat):
        raise CovarianceError(f"{label}_must_be_finite_numbers")
    for i in range(3):
        if matrix[i][i] < 0.0:
            raise CovarianceError(f"{label}_negative_variance")
        for j in range(i + 1, 3):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-8 * max(1.0, abs(matrix[i][j])):
                raise CovarianceError(f"{label}_not_symmetric")
    # 取对角为正的指标子块做正定性检查
    active = [i for i in range(3) if matrix[i][i] > 0.0]
    if len(active) >= 2:
        sub = [[matrix[i][j] for j in active] for i in active]
        try:
            cholesky(sub)
        except CovarianceError as exc:
            raise CovarianceError(f"{label}_not_positive_semidefinite") from exc


def slice_matrix(matrix3: list[list[float]], active: list[int]) -> list[list[float]]:
    return [[matrix3[i][j] for j in active] for i in active]


def simplex_project(values: list[float]) -> list[float]:
    """欧氏投影到 {f >= 0, sum f = 1}（Duchi 等，2008）。"""
    ordered = sorted(values, reverse=True)
    cumulative = 0.0
    rho = 0
    running: list[float] = []
    for j, value in enumerate(ordered):
        cumulative += value
        running.append(cumulative)
        if value - (cumulative - 1.0) / (j + 1) > 0.0:
            rho = j
    theta = (running[rho] - 1.0) / (rho + 1)
    return [max(0.0, value - theta) for value in values]


def jacobi_eigen(matrix: list[list[float]], max_sweeps: int = 100) -> tuple[list[float], list[list[float]]]:
    """实对称矩阵的 Jacobi 特征分解，返回 (升序特征值, 列特征向量)。"""
    n = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = max(
            (abs(a[i][j]) for i in range(n) for j in range(i + 1, n)),
            default=0.0,
        )
        if off <= 1e-14:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) <= 1e-14:
                    continue
                tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = 1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau))
                if tau < 0.0:
                    t = -t
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    vpk, vqk = v[k][p], v[k][q]
                    v[k][p] = c * vpk - s * vqk
                    v[k][q] = s * vpk + c * vqk
    eig = [(a[i][i], [v[k][i] for k in range(n)]) for i in range(n)]
    eig.sort(key=lambda item: item[0])
    values = [max(0.0, item[0]) for item in eig]
    vectors = [[item[1][k] for item in eig] for k in range(n)]
    return values, vectors


# ---------------------------------------------------------------- 反演求解

def solve_problem(
    observed: list[float],
    design: list[list[float]],
    obs_cov: list[list[float]],
    end_covs: list[list[list[float]]],
    max_iterations: int,
    tolerance: float,
) -> dict[str, Any]:
    """加权最小二乘混合反演。

    observed        长度 m 的观测值
    design          m x n，每行一个示踪指标，每列一个端元
    obs_cov         m x m 观测协方差（正定）
    end_covs        n 个 m x m 端元协方差（半正定）
    返回数值结果字典（不含名称等展示信息）。
    """
    m = len(observed)
    n = len(design[0])
    if m < 2:
        raise ValueError("insufficient_measurements")
    if len(design) != m or len(end_covs) != n:
        raise ValueError("dimension_mismatch")

    def covariance(frac: list[float]) -> list[list[float]]:
        cov = [row[:] for row in obs_cov]
        for j, fj in enumerate(frac):
            if fj == 0.0:
                continue
            weight = fj * fj
            ec = end_covs[j]
            for i in range(m):
                for k in range(m):
                    cov[i][k] += weight * ec[i][k]
        return cov

    def stats(frac: list[float]) -> tuple[list[float], list[list[float]], list[float], float]:
        residual = [
            sum(design[k][j] * frac[j] for j in range(n)) - observed[k]
            for k in range(m)
        ]
        cov = covariance(frac)
        u = solve_spd(cov, residual)
        phi = sum(residual[k] * u[k] for k in range(m))
        return residual, cov, u, phi

    fractions = [1.0 / n] * n
    step = 1.0
    residual, cov, u, phi = stats(fractions)
    recent = [phi]
    iterations = 0
    reason = "max_iterations"
    for iterations in range(1, max_iterations + 1):
        gradient = [
            2.0 * sum(design[k][j] * u[k] for k in range(m))
            - 2.0 * fractions[j] * quadratic(end_covs[j], u)
            for j in range(n)
        ]
        reference = max(recent)
        trial = None
        alpha = step
        for _ in range(60):
            candidate = simplex_project(
                [fractions[j] - alpha * gradient[j] for j in range(n)]
            )
            cand_residual, _, cand_u, cand_phi = stats(candidate)
            directional = sum(
                gradient[j] * (candidate[j] - fractions[j]) for j in range(n)
            )
            # 非单调 Armijo（窗口内最大值作参照），并要求数值上严格不上升
            if (
                cand_phi <= reference + 1e-4 * directional
                and cand_phi < phi * (1.0 + 1e-14) + 1e-14
            ):
                trial = (candidate, cand_phi, cand_u, alpha, directional, cand_residual)
                break
            alpha *= 0.5
        if trial is None:
            reason = "numerical_stall"
            break
        candidate, new_phi, new_u, alpha_used, directional, cand_residual = trial
        delta = [candidate[j] - fractions[j] for j in range(n)]
        new_gradient = [
            2.0 * sum(design[k][j] * new_u[k] for k in range(m))
            - 2.0 * candidate[j] * quadratic(end_covs[j], new_u)
            for j in range(n)
        ]
        # Barzilai-Borwein 步长，夹到合理量级，避免回溯后长期小步爬行
        curvature = sum(delta[j] * (new_gradient[j] - gradient[j]) for j in range(n))
        if abs(curvature) > 1e-30:
            step = min(
                max(sum(v * v for v in delta) / abs(curvature), 1e-10), 1e10
            )
        fractions = candidate
        phi = new_phi
        u = new_u
        residual = cand_residual
        recent.append(phi)
        if len(recent) > 10:
            recent.pop(0)
        # 投影梯度范数（实际位移 / 所用步长）作一阶平稳判据
        projected_gradient_norm = math.sqrt(sum(v * v for v in delta) / max(alpha_used * alpha_used, 1e-300))
        if (
            projected_gradient_norm <= 1e-9
            and abs(directional) <= tolerance * (1.0 + abs(phi))
        ):
            reason = "stationary"
            break

    residual, cov, u, phi = stats(fractions)
    sigma = [math.sqrt(max(0.0, cov[k][k])) for k in range(m)]
    predicted = [residual[k] + observed[k] for k in range(m)]
    z_score = [
        residual[k] / sigma[k] if sigma[k] > 0.0 else 0.0
        for k in range(m)
    ]
    chi_components = [residual[k] * u[k] for k in range(m)]

    dof = m - (n - 1)
    rank, condition, weakest_vector = _design_rank(design, cov, n)
    collinear = _collinear_pairs(design, cov)
    boundary = [j for j, fj in enumerate(fractions) if fj <= 1e-8]
    outlier_tracers = [k for k, z in enumerate(z_score) if abs(z) > OUTLIER_Z]
    reduced_chi = phi / dof if dof > 0 else None

    structural = m < n - 1
    identifiable = (
        not structural
        and rank == n - 1
        and math.isfinite(condition)
        and condition < ILL_CONDITIONED
    )

    return {
        "fractions": fractions,
        "predicted": predicted,
        "residual": residual,
        "sigma": sigma,
        "weighted_residual": z_score,
        "chi_components": chi_components,
        "chi_square": phi,
        "dof": dof,
        "reduced_chi_square": reduced_chi,
        "iterations": iterations,
        "converged": reason == "stationary",
        "convergence_reason": reason,
        "identifiable": identifiable,
        "effective_rank": rank,
        "condition_number": condition,
        "structural_underdetermined": structural,
        "collinear_pairs": collinear,
        "weakest_direction": weakest_vector,
        "boundary_endmembers": boundary,
        "outlier_tracers": outlier_tracers,
        "solution_covariance": cov,
    }


def _design_rank(
    design: list[list[float]],
    cov: list[list[float]],
    n_end: int,
) -> tuple[int, float, list[float]]:
    """以最后一个端元为参考构造加权差分矩阵 G，评估其有效秩。

    sum f = 1 时 y = a_n + sum_{j<n} f_j (a_j - a_n)，可辨识方向共 n-1 个。
    """
    m = len(design)
    dim = n_end - 1
    g = [
        [(design[k][j] - design[k][n_end - 1]) / math.sqrt(max(cov[k][k], 1e-30))
         for j in range(dim)]
        for k in range(m)
    ]
    gram = [
        [sum(g[k][i] * g[k][j] for k in range(m)) for j in range(dim)]
        for i in range(dim)
    ]
    values, vectors = jacobi_eigen(gram)
    largest = max(values, default=0.0)
    rank = sum(1 for value in values if value > largest * RANK_EPS)
    smallest = values[0] if values else 0.0
    floor = largest * RANK_EPS
    condition = largest / smallest if smallest > floor else largest / floor if largest > 0 else 1.0
    weakest = [vectors[k][0] for k in range(dim)] if vectors else []
    return rank, condition, weakest


def _collinear_pairs(
    design: list[list[float]],
    cov: list[list[float]],
) -> list[list[int]]:
    m = len(design)
    n_end = len(design[0])
    pairs: list[list[int]] = []
    for j in range(n_end):
        for ell in range(j + 1, n_end):
            dist2 = sum(
                (design[k][j] - design[k][ell]) ** 2 / max(cov[k][k], 1e-30)
                for k in range(m)
            )
            if dist2 < COLLINEAR_DIST2:
                pairs.append([j, ell])
    return pairs
