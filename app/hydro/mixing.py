"""受约束的加权最小二乘混合源反演。

模型（保守混合即质量守恒）::

    y = A f + e,   f_j >= 0, sum_j f_j = 1

每个观测指标 k 与端元 j 都可以携带标准差或完整协方差。端元特征
不确定度随混合比例传播，因此有效观测协方差为::

    C(f) = C_obs + sum_j f_j^2 * C_j

求解采用迭代重加权最小二乘（IRLS）：外层固定点迭代更新 C(f)，
内层在单纯形上解凸二次规划（原始活动集法，等式 KKT 直接求解），
因此比例始终严格非负且总和为一。

模块只依赖 Python 标准库，所有运算均为确定性运算：相同输入与
模型版本必然得到相同的比例数值与比例顺序。
"""

from __future__ import annotations

import math
from typing import Any

MODEL_NAME = "constrained-weighted-mixing"
MODEL_VERSION = "mix-wls-2.0"

TRACERS = ("isotope_d18o", "isotope_d2h", "solute_mg_l")
TRACER_LABELS = {
    "isotope_d18o": "δ18O",
    "isotope_d2h": "δ2H",
    "solute_mg_l": "溶质浓度",
}

_EPS = 2.220446049250313e-16
# 加权设计矩阵列共线判定阈值（相对最大列范数）
_COLINEAR_REL_TOL = 1e-10
# 判定端元是否参与混合（用于自由度与标准误）
_ACTIVE_FRACTION_TOL = 1e-8


class MixingError(ValueError):
    """反演输入或数值结构不可用。"""


# --------------------------------------------------------------------- 基础线性代数


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _matvec(a: list[list[float]], x: list[float]) -> list[float]:
    return [sum(a[i][j] * x[j] for j in range(len(x))) for i in range(len(a))]


def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gauss-Jordan 消元（部分主元）解线性方程组，奇异时抛出 MixingError。"""
    n = len(matrix)
    aug = [list(map(float, matrix[i])) + [float(rhs[i])] for i in range(n)]
    scale = max((abs(aug[i][j]) for i in range(n) for j in range(n)), default=0.0)
    pivot_tol = 1e-14 * max(1.0, scale)
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) <= pivot_tol:
            raise MixingError("singular_linear_system")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / pivot_value
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    columns = [_solve_linear(matrix, [1.0 if i == k else 0.0 for i in range(n)]) for k in range(n)]
    return [[columns[k][i] for k in range(n)] for i in range(n)]


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    """下三角 Cholesky 分解，非正定抛出 MixingError。"""
    n = len(matrix)
    lower = _zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if value <= 0.0:
                    raise MixingError("covariance_not_positive_definite")
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    return lower


def _forward_substitution(lower: list[list[float]], vector: list[float]) -> list[float]:
    """解 L x = b（L 下三角）。"""
    n = len(vector)
    out = [0.0] * n
    for i in range(n):
        out[i] = (vector[i] - sum(lower[i][k] * out[k] for k in range(i))) / lower[i][i]
    return out


# --------------------------------------------------------------------- 单纯形 QP


def _simplex_qp(
    hessian: list[list[float]],
    linear: list[float],
    max_iterations: int,
) -> tuple[list[float], str, float, int]:
    """解 min 0.5 f'Hf + q'f，约束 f>=0 且 1'f=1。

    原始活动集法：自由变量在等式约束上解 KKT 线性系统，越过下界
    的变量沿线搜索固定为 0，驻点处乘子违反的边界变量被释放。
    返回 (比例, 内层收敛原因, KKT 残差, 活动集迭代次数)。
    """
    n = len(linear)
    fractions = [1.0 / n] * n
    fixed: set[int] = set()
    h_scale = max(
        (abs(hessian[i][j]) for i in range(n) for j in range(n)),
        default=1.0,
    )
    bound_tol = 1e-11 * max(1.0, h_scale)
    multiplier_tol = 1e-10 * max(1.0, h_scale, max((abs(v) for v in linear), default=1.0))

    for iteration in range(max(1, max_iterations)):
        free = [i for i in range(n) if i not in fixed]
        rank = len(free)
        kkt = _zeros(rank + 1, rank + 1)
        rhs = [0.0] * (rank + 1)
        for a, idx in enumerate(free):
            for b, jdx in enumerate(free):
                kkt[a][b] = hessian[idx][jdx]
            kkt[a][rank] = 1.0
            kkt[rank][a] = 1.0
            rhs[a] = -linear[idx]
        rhs[rank] = 1.0
        solution = _solve_linear(kkt, rhs)
        lam = solution[rank]

        negative = [a for a in range(rank) if solution[a] < -bound_tol]
        if negative:
            step = 1.0
            candidates: list[tuple[float, int]] = []
            for a in negative:
                idx = free[a]
                current = fractions[idx]
                target = solution[a]
                candidate = current / (current - target)
                candidates.append((candidate, idx))
                if candidate < step - 1e-12:
                    step = candidate
            hitting = [idx for candidate, idx in candidates if abs(candidate - step) <= 1e-12]
            # 等式约束要求至少保留一个自由变量承载全部质量
            if len(hitting) >= rank:
                survivor = max(candidates, key=lambda item: (item[0], -item[1]))[1]
                hitting = [idx for idx in hitting if idx != survivor]
            for a, idx in enumerate(free):
                fractions[idx] += step * (solution[a] - fractions[idx])
            for idx in hitting:
                fractions[idx] = 0.0
                fixed.add(idx)
            continue

        for a, idx in enumerate(free):
            fractions[idx] = 0.0 if abs(solution[a]) < bound_tol else solution[a]
        gradient = _matvec(hessian, fractions)
        violations = [
            i for i in fixed if gradient[i] + linear[i] + lam < -multiplier_tol
        ]
        if not violations:
            # 清理线搜索累积的浮点舍入，使总和严格为一且不产生负比例
            drift = 1.0 - sum(fractions)
            if abs(drift) > 0.0:
                bearer = max(free, key=lambda idx: (fractions[idx], -idx))
                fractions[bearer] += drift
                if fractions[bearer] < 0.0:
                    fractions[bearer] = 0.0
            residual = _kkt_residual(hessian, linear, fractions, fixed, lam)
            return fractions, "converged", residual, iteration + 1
        # 释放最违反边界的变量（id 作为并列时的确定性次序）
        worst = min(violations, key=lambda i: (gradient[i] + linear[i] + lam, i))
        fixed.discard(worst)
    return fractions, "max_iterations", float("inf"), max_iterations


def _kkt_residual(
    hessian: list[list[float]],
    linear: list[float],
    fractions: list[float],
    fixed: set[int],
    lam: float,
) -> float:
    gradient = _matvec(hessian, fractions)
    return max(
        (abs(gradient[i] + linear[i] + lam) for i in range(len(fractions)) if i not in fixed),
        default=0.0,
    )


# --------------------------------------------------------------------- 诊断


def _pivoted_column_rank(
    matrix: list[list[float]],
) -> tuple[int, list[int], list[int]]:
    """对 B（m×n）做列主元修正 Gram-Schmidt。

    返回 (秩, 选入基的列序, 被判定为近共线的列)。
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    vectors = [[matrix[i][j] for i in range(rows)] for j in range(cols)]
    initial_scale = max((math.sqrt(sum(v * v for v in col)) for col in vectors), default=0.0)
    tolerance = _COLINEAR_REL_TOL * max(1.0, initial_scale)
    chosen: list[int] = []
    rejected: list[int] = []
    remaining = set(range(cols))
    while remaining:
        pivot = max(remaining, key=lambda j: (math.sqrt(sum(v * v for v in vectors[j])), -j))
        norm = math.sqrt(sum(v * v for v in vectors[pivot]))
        if norm < tolerance:
            rejected.extend(sorted(remaining))
            break
        unit = [v / norm for v in vectors[pivot]]
        chosen.append(pivot)
        remaining.discard(pivot)
        for j in remaining:
            projection = sum(unit[i] * vectors[j][i] for i in range(rows))
            vectors[j] = [vectors[j][i] - projection * unit[i] for i in range(rows)]
    return len(chosen), chosen, sorted(rejected)


def _fraction_standard_errors(
    design: list[list[float]],
    weights: list[list[float]],
    fractions: list[float],
) -> list[float | None]:
    """活动端元比例的近似标准误。

    Fisher 信息 G = A_S' W A_S；在等式约束 1'f=1 下的协方差为
    KKT 系统 [[G,1],[1',0]]^{-1} 的左上块（Schur 补）。系统奇异
    （如非零端元多于可辨识自由度）时对应项返回 None。
    """
    active = [j for j, f in enumerate(fractions) if f > _ACTIVE_FRACTION_TOL]
    result: list[float | None] = [None] * len(fractions)
    dimension = len(active)
    if dimension == 1:
        result[active[0]] = 0.0
        return result
    gram = _zeros(dimension, dimension)
    for a, ja in enumerate(active):
        for b, jb in enumerate(active):
            gram[a][b] = sum(
                design[k][ja] * weights[k][l] * design[l][jb]
                for k in range(len(design))
                for l in range(len(design))
            )
    kkt = _zeros(dimension + 1, dimension + 1)
    for a in range(dimension):
        for b in range(dimension):
            kkt[a][b] = gram[a][b]
        kkt[a][dimension] = 1.0
        kkt[dimension][a] = 1.0
    try:
        for a in range(dimension):
            rhs = [0.0] * (dimension + 1)
            rhs[a] = 1.0
            column = _solve_linear(kkt, rhs)
            value = column[a]
            result[active[a]] = math.sqrt(value) if value > 0.0 else 0.0
    except MixingError:
        return result
    return result


# --------------------------------------------------------------------- 主求解流程


def _effective_covariance(
    fractions: list[float],
    obs_sigmas: list[float],
    endmember_sigmas: list[list[float]],
    obs_cov: list[list[float]] | None,
    endmember_covs: list[list[float] | None],
    active: list[int],
) -> list[list[float]]:
    m = len(active)
    cov = _zeros(m, m)
    if obs_cov is not None:
        for a, k in enumerate(active):
            for b, l in enumerate(active):
                cov[a][b] += obs_cov[k][l]
    else:
        for a, k in enumerate(active):
            cov[a][a] += obs_sigmas[k] ** 2
    for j, fraction in enumerate(fractions):
        weight = fraction * fraction
        if weight <= 0.0:
            continue
        end_cov = endmember_covs[j]
        if end_cov is not None:
            for a, k in enumerate(active):
                for b, l in enumerate(active):
                    cov[a][b] += weight * end_cov[k][l]
        else:
            for a, k in enumerate(active):
                cov[a][a] += weight * endmember_sigmas[j][k] ** 2
    return cov


def _with_jitter(cov: list[list[float]], level: float) -> list[list[float]]:
    m = len(cov)
    out = [row[:] for row in cov]
    for i in range(m):
        out[i][i] += level
    return out


def solve_weighted_mixture(
    observed: list[float | None],
    endmembers: list[dict[str, Any]],
    weights: dict[str, Any],
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
    irls_max_iterations: int = 50,
    method: str = "weighted-least-squares",
    model_version: str = "mix-1",
) -> dict[str, Any]:
    """执行带不确定度的混合反演。

    ``observed`` 为按 :data:`TRACERS` 顺序排列的观测值（缺失为 None）；
    ``endmembers`` 每项含 id/name/signatures(3)/sigmas(3)/covariance；
    ``weights`` 由服务层解析，包含 observation_sigmas、observation_covariance、
    endmember_sigmas 与 endmember_covariances。
    """
    if len(observed) != len(TRACERS):
        raise MixingError("unexpected_tracer_count")
    active = [k for k, value in enumerate(observed) if value is not None]
    if len(active) < 2:
        raise MixingError("insufficient_measurements")
    if len(endmembers) < 2:
        raise MixingError("insufficient_endmembers")

    n = len(endmembers)
    m = len(active)
    obs_sigmas = [max(float(weights["observation_sigmas"].get(TRACERS[k], 0.0) or 0.0), 1e-12) for k in range(3)]
    obs_cov = weights.get("observation_covariance")
    endmember_sigmas = [
        [float(endmembers[j]["sigmas"][TRACERS[k]]) for k in range(3)] for j in range(n)
    ]
    endmember_covs = [endmembers[j].get("covariance") for j in range(n)]

    design_full = [[float(endmembers[j]["signatures"][TRACERS[k]]) for j in range(n)] for k in range(3)]
    design = [[design_full[k][j] for j in range(n)] for k in active]
    targets = [float(observed[k]) for k in active]

    fractions = [1.0 / n] * n
    jitter_level = 0.0
    outer_reason = "converged"
    last_change = float("inf")
    qp_solves = 0
    total_inner_iterations = 0

    def build_solve(frac: list[float]) -> tuple[list[list[float]], list[list[float]], list[float], str, float, float, int]:
        cov = _effective_covariance(
            frac, obs_sigmas, endmember_sigmas, obs_cov, endmember_covs, active
        )
        diagonal_scale = max((cov[i][i] for i in range(m)), default=1.0)
        level = 0.0
        for _ in range(6):
            candidate = cov if level == 0.0 else _with_jitter(cov, level)
            try:
                lower = _cholesky(candidate)
                weighted = _inverse(candidate)
                break
            except MixingError:
                level = max(1e-12 * max(1.0, diagonal_scale), level * 100.0)
        else:
            raise MixingError("covariance_not_positive_definite")
        hessian = _zeros(n, n)
        for i in range(n):
            for j in range(n):
                hessian[i][j] = 2.0 * sum(
                    design[k][i] * weighted[k][l] * design[l][j]
                    for k in range(m)
                    for l in range(m)
                )
        # 端元特征共线时 Hessian 在等式约束零空间上半正定但奇异，
        # 加入仅作用于零空间方向的微小脊（1 方向无扰动，不改变总和约束）
        h_scale = max((hessian[i][i] for i in range(n)), default=1.0)
        ridge = 1e-11 * max(1.0, h_scale)
        centering = [[(1.0 if i == j else 0.0) - 1.0 / n for j in range(n)] for i in range(n)]
        regularized = [
            [hessian[i][j] + ridge * centering[i][j] for j in range(n)] for i in range(n)
        ]
        linear = [
            -2.0 * sum(design[k][i] * weighted[k][l] * targets[l] for k in range(m) for l in range(m))
            for i in range(n)
        ]
        new_frac, inner_reason, residual, inner_iterations = _simplex_qp(regularized, linear, max_iterations)
        return weighted, candidate, new_frac, inner_reason, level, residual, inner_iterations

    for outer in range(max(1, irls_max_iterations)):
        weighted, cov, new_fractions, inner_reason, level, _, inner_iterations = build_solve(fractions)
        qp_solves += 1
        total_inner_iterations += inner_iterations
        jitter_level = max(jitter_level, level)
        last_change = max(abs(new_fractions[j] - fractions[j]) for j in range(n))
        fractions = new_fractions
        if inner_reason != "converged":
            outer_reason = inner_reason
        if last_change < tolerance and inner_reason == "converged":
            outer_reason = "converged"
            break
    else:
        outer_reason = "max_iterations"

    # 用最终比例再组装一次协方差并求解，保证权重与比例自洽
    weighted, covariance, final_fractions, final_reason, level, kkt_residual, final_inner_iterations = build_solve(fractions)
    qp_solves += 1
    total_inner_iterations += final_inner_iterations
    jitter_level = max(jitter_level, level)
    final_change = max(abs(final_fractions[j] - fractions[j]) for j in range(n))
    fractions = final_fractions
    if final_reason != "converged":
        convergence_reason = final_reason
    elif jitter_level >= 1e-8 * max(1.0, max((covariance[i][i] for i in range(m)), default=1.0)):
        convergence_reason = "numerical_degeneracy"
    elif outer_reason == "max_iterations" and final_change > tolerance:
        convergence_reason = "max_iterations"
    else:
        convergence_reason = "converged"

    predicted = _matvec(design_full, fractions)
    residuals_active = [predicted[k] - float(observed[k]) for k in active]
    lower = _cholesky(covariance)
    whitened = _forward_substitution(lower, residuals_active)
    chi_square = sum(v * v for v in whitened)

    # 可辨识性诊断：加权设计矩阵 B = L^{-1} A
    whitened_design = [
        _forward_substitution(lower, [design[k][j] for k in range(m)]) for j in range(n)
    ]
    # 转置为 m×n
    basis = [[whitened_design[j][k] for j in range(n)] for k in range(m)]
    rank, _chosen, collinear = _pivoted_column_rank(basis)
    standard_errors = _fraction_standard_errors(design, weighted, fractions)
    active_sources = sum(1 for f in fractions if f > _ACTIVE_FRACTION_TOL)
    dof = m - active_sources
    reduced_chi_square = chi_square / dof if dof > 0 else None

    warnings_list: list[dict[str, Any]] = []
    if m < n:
        warnings_list.append({
            "code": "observation_count_below_endmember_count",
            "endmembers": [],
            "detail": f"活跃观测指标 {m} 个少于补给端元 {n} 个，比例不唯一。",
        })
    if rank < n:
        warnings_list.append({
            "code": "collinear_endmember_signatures",
            "endmembers": [endmembers[j]["name"] for j in collinear],
            "detail": "部分端元在加权观测空间中的特征近共线，无法独立辨识。",
        })
    if active_sources > m:
        weak_names = [
            endmembers[j]["name"]
            for j in range(n)
            if fractions[j] > _ACTIVE_FRACTION_TOL
        ]
        warnings_list.append({
            "code": "more_active_sources_than_observations",
            "endmembers": weak_names,
            "detail": "非零比例端元数超过活跃观测指标数，存在等价混合方案。",
        })
    poorly = [
        endmembers[j]["name"]
        for j in range(n)
        if (standard_errors[j] is not None and standard_errors[j] > 0.1)
        or (standard_errors[j] is None and fractions[j] > _ACTIVE_FRACTION_TOL)
    ]
    if poorly:
        warnings_list.append({
            "code": "weakly_constrained_fractions",
            "endmembers": poorly,
            "detail": "这些端元比例的近似标准误偏大（>0.1）或无法估计。",
        })
    if reduced_chi_square is not None and reduced_chi_square > 4.0:
        warnings_list.append({
            "code": "large_weighted_residuals",
            "endmembers": [],
            "detail": f"约化卡方 {reduced_chi_square:.3g} 明显偏大，观测与保守混合模型不自洽。",
        })

    unidentifiable_codes = {
        "observation_count_below_endmember_count",
        "collinear_endmember_signatures",
    }
    if any(w["code"] in unidentifiable_codes for w in warnings_list):
        identifiability_status = "unidentifiable"
    elif warnings_list:
        identifiability_status = "weak"
    else:
        identifiability_status = "identified"

    # 比例顺序确定性：按比例降序；9 位小数内视为并列时名次相同，次序按端元 id 升序
    order = sorted(range(n), key=lambda j: (-round(fractions[j], 9), endmembers[j]["id"]))
    ranks = [0] * n
    position = 0
    while position < n:
        idx = order[position]
        end = position + 1
        while end < n and round(fractions[order[end]], 9) == round(fractions[idx], 9):
            end += 1
        for k in range(position, end):
            ranks[order[k]] = position + 1
        position = end

    def clean(value: float, digits: int = 10) -> float:
        result = round(value, digits)
        return 0.0 if result == 0.0 else result

    residual_rows = []
    for a, k in enumerate(active):
        sigma = math.sqrt(covariance[a][a])
        residual_rows.append({
            "tracer": TRACERS[k],
            "tracer_label": TRACER_LABELS[TRACERS[k]],
            "observed": observed[k],
            "predicted": clean(predicted[k], 8),
            "residual": clean(residuals_active[a], 8),
            "sigma": clean(sigma, 8),
            "standardized_residual": clean(residuals_active[a] / sigma if sigma > 0.0 else 0.0, 8),
            "active": True,
        })
    for k in range(3):
        if k not in active:
            residual_rows.append({
                "tracer": TRACERS[k],
                "tracer_label": TRACER_LABELS[TRACERS[k]],
                "observed": None,
                "predicted": clean(predicted[k], 8),
                "residual": None,
                "sigma": None,
                "standardized_residual": None,
                "active": False,
            })

    fractions_rows = []
    for j, endmember in enumerate(endmembers):
        fractions_rows.append({
            "endmember_id": endmember["id"],
            "name": endmember["name"],
            "fraction": clean(fractions[j], 10),
            "rank": ranks[j],
            "standard_error": clean(standard_errors[j], 8) if standard_errors[j] is not None else None,
        })

    return {
        "fractions": fractions_rows,
        "mass_balance": clean(sum(fractions), 12),
        "predicted": {TRACERS[k]: clean(predicted[k], 8) for k in range(3)},
        "residuals": residual_rows,
        "weighted_residuals": [clean(v, 8) for v in whitened],
        "chi_square": clean(chi_square, 8),
        "reduced_chi_square": clean(reduced_chi_square, 8) if reduced_chi_square is not None else None,
        "degrees_of_freedom": dof if dof > 0 else None,
        "weighted_rmse": clean(math.sqrt(chi_square / m), 8),
        "rmse": clean(math.sqrt(chi_square / m), 8),
        "effective_covariance": [[clean(covariance[a][b], 10) for b in range(m)] for a in range(m)],
        "iterations": outer + 1,
        "irls_iterations": outer + 1,
        "qp_solves": qp_solves,
        "active_set_iterations": total_inner_iterations,
        "kkt_residual": clean(kkt_residual, 12),
        "converged": convergence_reason == "converged",
        "convergence_reason": convergence_reason,
        "identifiability": {
            "status": identifiability_status,
            "warnings": warnings_list,
            "design_rank": rank,
            "observations": m,
            "active_endmember_count": active_sources,
            "endmember_count": n,
        },
        "model_summary": {
            "name": MODEL_NAME,
            "solver_version": MODEL_VERSION,
            "model_version": model_version,
            "method": method,
            "objective": "weighted_chi_square",
            "constraints": ["fractions_non_negative", "fractions_sum_to_one", "conservative_mass_balance"],
            "uncertainty_propagation": "C(f)=C_obs+sum(f_j^2*C_j)",
            "active_tracers": [TRACERS[k] for k in active],
            "max_iterations": max_iterations,
            "irls_max_iterations": irls_max_iterations,
            "tolerance": tolerance,
            "jitter_applied": jitter_level > 0.0,
            "nullspace_ridge_applied": rank < n,
            "deterministic": True,
        },
    }
