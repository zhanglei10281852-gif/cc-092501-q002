from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.database import get_connection, transaction
from app.hydro import mixing


SCHEMA = """
CREATE TABLE IF NOT EXISTS hydro_wells (
 id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
 latitude REAL NOT NULL, longitude REAL NOT NULL, aquifer TEXT NOT NULL, screen_depth_m REAL NOT NULL,
 status TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hydro_endmembers (
 id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, isotope_d18o REAL NOT NULL,
 isotope_d2h REAL NOT NULL, solute_mg_l REAL NOT NULL, uncertainty REAL NOT NULL,
 isotope_d18o_std REAL, isotope_d2h_std REAL, solute_std REAL,
 covariance_json TEXT NOT NULL DEFAULT 'null',
 version TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)), created_at TEXT NOT NULL,
 UNIQUE(name,version)
);
CREATE TABLE IF NOT EXISTS hydro_samples (
 id INTEGER PRIMARY KEY AUTOINCREMENT, well_id INTEGER NOT NULL REFERENCES hydro_wells(id) ON DELETE RESTRICT,
 sample_code TEXT NOT NULL UNIQUE, sampled_at TEXT NOT NULL, isotope_d18o REAL, isotope_d2h REAL,
 solute_mg_l REAL, detection_limit REAL NOT NULL, measurement_error REAL NOT NULL,
 isotope_d18o_std REAL, isotope_d2h_std REAL, solute_std REAL,
 covariance_json TEXT NOT NULL DEFAULT 'null',
 quality_status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hydro_inversions (
 id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id INTEGER NOT NULL REFERENCES hydro_samples(id) ON DELETE RESTRICT,
 task_key TEXT NOT NULL UNIQUE, model_version TEXT NOT NULL, method TEXT NOT NULL,
 input_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
 worker_id TEXT NOT NULL DEFAULT '', result_json TEXT NOT NULL DEFAULT '{}', error TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hydro_transport_runs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, well_id INTEGER NOT NULL REFERENCES hydro_wells(id) ON DELETE RESTRICT,
 task_key TEXT NOT NULL UNIQUE, model_version TEXT NOT NULL, input_json TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'done', result_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hydro_audit (
 id INTEGER PRIMARY KEY AUTOINCREMENT, resource_type TEXT NOT NULL, resource_id INTEGER,
 action TEXT NOT NULL, actor TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hydro_samples_well ON hydro_samples(well_id,sampled_at);
CREATE INDEX IF NOT EXISTS idx_hydro_inversions_status ON hydro_inversions(status,created_at);
"""

# 旧库增量列：表名 -> [(列名, 列定义)]
_ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "hydro_endmembers": [
        ("isotope_d18o_std", "REAL"),
        ("isotope_d2h_std", "REAL"),
        ("solute_std", "REAL"),
        ("covariance_json", "TEXT NOT NULL DEFAULT 'null'"),
    ],
    "hydro_samples": [
        ("isotope_d18o_std", "REAL"),
        ("isotope_d2h_std", "REAL"),
        ("solute_std", "REAL"),
        ("covariance_json", "TEXT NOT NULL DEFAULT 'null'"),
    ],
}

_STD_FLOOR = 1e-12


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_schema() -> None:
    connection = get_connection()
    connection.executescript(SCHEMA)
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        for name, definition in columns:
            if name not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


def _std_vector(row: dict[str, Any], explicit_cols: tuple[str, str, str], fallback: float) -> list[float]:
    """逐指标 1σ：优先逐指标列，其次统一兜底值；零值抬到数值下限。"""
    values = []
    for col in explicit_cols:
        value = row.get(col)
        if value is None:
            value = fallback
        values.append(max(float(value), _STD_FLOOR))
    return values


def _diag3(std: list[float]) -> list[list[float]]:
    return [[std[i] ** 2 if i == j else 0.0 for j in range(3)] for i in range(3)]


def _covariance3(row: dict[str, Any], std_cols: tuple[str, str, str], fallback: float) -> list[list[float]]:
    """取行记录的 3x3 协方差；未提供完整协方差时用逐指标标准差生成对角阵。"""
    stored = row.get("covariance_json")
    matrix = json.loads(stored) if stored else None
    if matrix is not None:
        mixing.validate_covariance3(matrix, "endmember" if "uncertainty" in row else "observation")
        return matrix
    return _diag3(_std_vector(row, std_cols, fallback))


class HydroService:
    def __init__(self, connection: sqlite3.Connection | None = None):
        self.connection = connection or get_connection()
        ensure_schema()

    def create_well(self, payload: dict[str, Any], actor: str = "researcher") -> dict[str, Any]:
        now = _now()
        with transaction(immediate=True) as connection:
            cursor = connection.execute("INSERT INTO hydro_wells(code,name,latitude,longitude,aquifer,screen_depth_m,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", (payload["code"],payload["name"],payload["latitude"],payload["longitude"],payload["aquifer"],payload["screen_depth_m"],now,now))
            well_id = cursor.lastrowid
            connection.execute("INSERT INTO hydro_audit(resource_type,resource_id,action,actor,payload_json,created_at) VALUES('well',?,?,?,?,?)", (well_id,"create",actor,json.dumps(payload,ensure_ascii=False),now))
            return dict(connection.execute("SELECT * FROM hydro_wells WHERE id=?",(well_id,)).fetchone())

    def get_well(self, well_id: int) -> dict[str, Any] | None:
        well = self.connection.execute("SELECT * FROM hydro_wells WHERE id=?",(well_id,)).fetchone()
        if well is None: return None
        result = dict(well)
        result["samples"] = [dict(r) for r in self.connection.execute("SELECT * FROM hydro_samples WHERE well_id=? ORDER BY sampled_at,id",(well_id,)).fetchall()]
        return result

    def delete_well(self, well_id: int) -> bool:
        with transaction(immediate=True) as connection:
            cursor = connection.execute("DELETE FROM hydro_wells WHERE id=?",(well_id,))
            if cursor.rowcount == 0: raise KeyError("well_not_found")
            return True

    def create_endmember(self, payload: dict[str, Any]) -> dict[str, Any]:
        now=_now()
        covariance = json.dumps(payload["covariance"], ensure_ascii=False) if payload.get("covariance") is not None else "null"
        with transaction(immediate=True) as connection:
            cursor=connection.execute(
                "INSERT INTO hydro_endmembers(name,isotope_d18o,isotope_d2h,solute_mg_l,uncertainty,"
                "isotope_d18o_std,isotope_d2h_std,solute_std,covariance_json,version,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (payload["name"],payload["isotope_d18o"],payload["isotope_d2h"],payload["solute_mg_l"],
                 payload["uncertainty"],payload.get("isotope_d18o_std"),payload.get("isotope_d2h_std"),
                 payload.get("solute_std"),covariance,payload["version"],now))
            return dict(connection.execute("SELECT * FROM hydro_endmembers WHERE id=?",(cursor.lastrowid,)).fetchone())

    def add_sample(self, well_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        if self.connection.execute("SELECT id FROM hydro_wells WHERE id=?",(well_id,)).fetchone() is None: raise KeyError("well_not_found")
        values=[payload.get("isotope_d18o"),payload.get("isotope_d2h"),payload.get("solute_mg_l")]
        quality="usable" if sum(v is not None for v in values)>=2 else "incomplete"
        covariance = json.dumps(payload["covariance"], ensure_ascii=False) if payload.get("covariance") is not None else "null"
        now=_now()
        with transaction(immediate=True) as connection:
            cursor=connection.execute(
                "INSERT INTO hydro_samples(well_id,sample_code,sampled_at,isotope_d18o,isotope_d2h,solute_mg_l,"
                "detection_limit,measurement_error,isotope_d18o_std,isotope_d2h_std,solute_std,covariance_json,"
                "quality_status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (well_id,payload["sample_code"],payload["sampled_at"],payload.get("isotope_d18o"),
                 payload.get("isotope_d2h"),payload.get("solute_mg_l"),payload["detection_limit"],
                 payload["measurement_error"],payload.get("isotope_d18o_std"),payload.get("isotope_d2h_std"),
                 payload.get("solute_std"),covariance,quality,now))
            return dict(connection.execute("SELECT * FROM hydro_samples WHERE id=?",(cursor.lastrowid,)).fetchone())

    # ------------------------------------------------------------ 混合反演

    def _build_problem(self, sample: dict[str, Any], endmembers: list[dict[str, Any]]) -> dict[str, Any]:
        """从（快照中的）样本与端元记录组装加权最小二乘问题。"""
        observed_all = [sample.get(t) for t in mixing.TRACERS]
        active = [k for k, value in enumerate(observed_all) if value is not None]
        if len(active) < 2:
            raise ValueError("insufficient_measurements")

        obs_cov_full = _covariance3(
            sample,
            ("isotope_d18o_std", "isotope_d2h_std", "solute_std"),
            max(float(sample.get("measurement_error") or 0.0), _STD_FLOOR),
        )
        end_covs_full = [
            _covariance3(
                e,
                ("isotope_d18o_std", "isotope_d2h_std", "solute_std"),
                float(e["uncertainty"]),
            )
            for e in endmembers
        ]
        for cov, who in [(obs_cov_full, "observation")] + [(c, "endmember") for c in end_covs_full]:
            mixing.validate_covariance3(cov, who)

        observed = [float(observed_all[k]) for k in active]
        design = [[float(e[mixing.TRACERS[k]]) for e in endmembers] for k in active]
        obs_cov = mixing.slice_matrix(obs_cov_full, active)
        end_covs = [mixing.slice_matrix(c, active) for c in end_covs_full]
        # 活动子块必须正定，否则求解无意义
        mixing.cholesky(obs_cov)
        return {
            "active": active,
            "observed": observed,
            "design": design,
            "obs_cov": obs_cov,
            "obs_cov_full": obs_cov_full,
            "end_covs": end_covs,
            "end_covs_full": end_covs_full,
        }

    def solve_mixture(
        self,
        sample: dict[str, Any],
        endmembers: list[dict[str, Any]],
        max_iterations: int,
        tolerance: float,
        model_version: str,
        method: str,
    ) -> dict[str, Any]:
        problem = self._build_problem(sample, endmembers)
        active = problem["active"]
        tracers = [mixing.TRACERS[k] for k in active]
        solution = mixing.solve_problem(
            problem["observed"], problem["design"], problem["obs_cov"],
            problem["end_covs"], max_iterations, tolerance,
        )

        fractions_raw = solution["fractions"]
        order = sorted(range(len(endmembers)), key=lambda j: (-fractions_raw[j], endmembers[j]["id"]))
        ranks = [0] * len(endmembers)
        for rank, j in enumerate(order, start=1):
            ranks[j] = rank
        fractions = [
            {
                "endmember_id": endmembers[j]["id"],
                "name": endmembers[j]["name"],
                "fraction": round(fractions_raw[j], 8),
                "rank": ranks[j],
            }
            for j in range(len(endmembers))
        ]

        n_end = len(endmembers)
        warnings: list[dict[str, str]] = []
        if solution["structural_underdetermined"]:
            warnings.append({"code": "structural_underdetermined",
                             "message": f"有效示踪指标 {len(active)} 个少于端元自由度 {n_end - 1}，比例无唯一解"})
        if solution["effective_rank"] < n_end - 1:
            warnings.append({"code": "rank_deficient",
                             "message": "端元在加权示踪空间中线性相关，存在无法区分的补给组合"})
        elif solution["condition_number"] >= mixing.ILL_CONDITIONED:
            warnings.append({"code": "ill_conditioned",
                             "message": f"设计矩阵条件数约 {solution['condition_number']:.2e}，比例对测量误差极度敏感"})
        for pair in solution["collinear_pairs"]:
            warnings.append({"code": "collinear_endmembers",
                             "message": f"端元 {endmembers[pair[0]]['name']} 与 {endmembers[pair[1]]['name']} 示踪特征近似重合，难以区分"})
        for j in solution["boundary_endmembers"]:
            warnings.append({"code": "zero_fraction_boundary",
                             "message": f"端元 {endmembers[j]['name']} 最优比例为 0（贴单纯形边界），其贡献未获数据支持"})
        for k in solution["outlier_tracers"]:
            warnings.append({"code": "residual_outlier",
                             "message": f"指标 {tracers[k]} 加权残差绝对值超过 {mixing.OUTLIER_Z:.0f}σ，建议核查测量或端元代表性"})
        if not solution["converged"]:
            warnings.append({"code": "not_converged",
                             "message": f"优化未平稳（{solution['convergence_reason']}），结果可能只是迭代停止点"})

        obs_records = []
        for idx, k in enumerate(active):
            obs_records.append({
                "tracer": mixing.TRACERS[k],
                "observed": round(problem["observed"][idx], 10),
                "predicted": round(solution["predicted"][idx], 10),
                "residual": round(solution["residual"][idx], 10),
                "sigma": round(solution["sigma"][idx], 10),
                "weighted_residual": round(solution["weighted_residual"][idx], 8),
                "chi_square_component": round(solution["chi_components"][idx], 8),
            })

        weights = {
            "tracers": list(mixing.TRACERS),
            "active_tracers": tracers,
            "observation": {
                "std": [math.sqrt(problem["obs_cov_full"][k][k]) for k in range(3)],
                "covariance": problem["obs_cov_full"],
            },
            "endmembers": [
                {
                    "endmember_id": endmembers[j]["id"],
                    "name": endmembers[j]["name"],
                    "std": [math.sqrt(problem["end_covs_full"][j][k][k]) for k in range(3)],
                    "covariance": problem["end_covs_full"][j],
                }
                for j in range(n_end)
            ],
            "covariance_model": "C(f)=C_obs+sum_j f_j^2 C_j",
        }

        return {
            "fractions": fractions,
            "fraction_order": [endmembers[j]["id"] for j in order],
            "mass_balance": round(sum(fractions_raw), 10),
            "observations": obs_records,
            "weighted_residuals": [record["weighted_residual"] for record in obs_records],
            "chi_square": round(solution["chi_square"], 10),
            "degrees_of_freedom": solution["dof"],
            "reduced_chi_square": None if solution["reduced_chi_square"] is None
            else round(solution["reduced_chi_square"], 8),
            "weighted_rms": round(math.sqrt(solution["chi_square"] / len(active)), 8),
            "iterations": solution["iterations"],
            "converged": solution["converged"],
            "convergence_reason": solution["convergence_reason"],
            "identifiable": solution["identifiable"],
            "warnings": warnings,
            "diagnostics": {
                "effective_rank": solution["effective_rank"],
                "required_rank": n_end - 1,
                "condition_number": None if not math.isfinite(solution["condition_number"])
                else round(solution["condition_number"], 6),
                "weakest_direction": [round(v, 8) for v in solution["weakest_direction"]],
                "collinear_pairs": solution["collinear_pairs"],
                "boundary_endmember_ids": [endmembers[j]["id"] for j in solution["boundary_endmembers"]],
                "outlier_tracers": [tracers[k] for k in solution["outlier_tracers"]],
            },
            "weights": weights,
            "model": {
                "model_version": model_version,
                "method": method,
                "solver_version": mixing.SOLVER_VERSION,
                "objective": "chi_square",
                "constraints": ["fractions_non_negative", "fractions_sum_to_one"],
            },
        }

    def enqueue_inversion(self, sample_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        sample=self.connection.execute("SELECT * FROM hydro_samples WHERE id=?",(sample_id,)).fetchone()
        if sample is None: raise KeyError("sample_not_found")
        ids=sorted(set(payload["endmember_ids"]))
        endmembers=self.connection.execute(f"SELECT * FROM hydro_endmembers WHERE active=1 AND id IN ({','.join('?' for _ in ids)}) ORDER BY id",ids).fetchall()
        if len(endmembers)!=len(ids): raise ValueError("endmember_not_found")
        # 输入快照包含原始观测、逐指标标准差与协方差，是任务去重与日后复算的唯一依据
        input_data={**payload,"endmember_ids":ids,"sample":dict(sample),"endmembers":[dict(e) for e in endmembers]}
        key=_digest(input_data); now=_now()
        with transaction(immediate=True) as connection:
            old=connection.execute("SELECT * FROM hydro_inversions WHERE task_key=?",(key,)).fetchone()
            if old: return dict(old)
            cursor=connection.execute("INSERT INTO hydro_inversions(sample_id,task_key,model_version,method,input_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",(sample_id,key,payload["model_version"],payload["method"],json.dumps(input_data,ensure_ascii=False),now,now))
            return dict(connection.execute("SELECT * FROM hydro_inversions WHERE id=?",(cursor.lastrowid,)).fetchone())

    def get_inversion(self, task_id: int) -> dict[str, Any] | None:
        task = self.connection.execute("SELECT * FROM hydro_inversions WHERE id=?", (task_id,)).fetchone()
        return dict(task) if task is not None else None

    def run_inversion(self, task_id: int, worker_id: str) -> dict[str, Any]:
        with transaction(immediate=True) as connection:
            task=connection.execute("SELECT * FROM hydro_inversions WHERE id=?",(task_id,)).fetchone()
            if task is None: raise KeyError("task_not_found")
            if task["status"]=="done": return dict(task)
            connection.execute("UPDATE hydro_inversions SET status='running',attempts=attempts+1,worker_id=?,updated_at=? WHERE id=?",(worker_id,_now(),task_id))
        data=json.loads(task["input_json"])
        # 仅依据入队时固化的快照求解：端元库后续改动不影响历史任务的复算结果
        try:
            result=self.solve_mixture(
                data["sample"], data["endmembers"],
                data["max_iterations"], data["tolerance"],
                data["model_version"], data["method"],
            )
        except Exception as exc:
            with transaction(immediate=True) as connection: connection.execute("UPDATE hydro_inversions SET status='failed',error=?,updated_at=? WHERE id=?",(str(exc),_now(),task_id))
            raise
        with transaction(immediate=True) as connection:
            connection.execute("UPDATE hydro_inversions SET status='done',result_json=?,error='',updated_at=? WHERE id=?",(json.dumps(result,ensure_ascii=False),_now(),task_id))
            return dict(connection.execute("SELECT * FROM hydro_inversions WHERE id=?",(task_id,)).fetchone())

    def run_transport(self, well_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        if self.connection.execute("SELECT id FROM hydro_wells WHERE id=?",(well_id,)).fetchone() is None: raise KeyError("well_not_found")
        key=_digest({"well_id":well_id,**payload}); now=_now()
        old=self.connection.execute("SELECT * FROM hydro_transport_runs WHERE task_key=?",(key,)).fetchone()
        if old: return dict(old)
        points=[]; t=payload["step_days"]
        while t<=payload["duration_days"]+1e-12:
            d=payload["dispersion_m2_day"]; x=payload["distance_m"]; v=payload["velocity_m_day"]
            c=payload["source_concentration"]*math.exp(-((x-v*t)**2)/(4*d*t))*math.exp(-payload["decay_per_day"]*t)/math.sqrt(4*math.pi*d*t)
            points.append({"time_days":round(t,8),"concentration":c}); t+=payload["step_days"]
        peak=max(points,key=lambda p:p["concentration"])
        result={"points":points,"peak":peak,"arrival_time_days":payload["distance_m"]/payload["velocity_m_day"],"model_version":payload["model_version"]}
        with transaction(immediate=True) as connection:
            cursor=connection.execute("INSERT INTO hydro_transport_runs(well_id,task_key,model_version,input_json,status,result_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",(well_id,key,payload["model_version"],json.dumps(payload,ensure_ascii=False),"done",json.dumps(result,ensure_ascii=False),now,now))
            return dict(connection.execute("SELECT * FROM hydro_transport_runs WHERE id=?",(cursor.lastrowid,)).fetchone())
