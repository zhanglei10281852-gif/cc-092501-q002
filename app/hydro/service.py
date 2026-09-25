from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.database import get_connection, transaction
from app.hydro.mixing import MODEL_VERSION as SOLVER_VERSION
from app.hydro.mixing import TRACERS, MixingError, solve_weighted_mixture


SCHEMA = """
CREATE TABLE IF NOT EXISTS hydro_wells (
 id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
 latitude REAL NOT NULL, longitude REAL NOT NULL, aquifer TEXT NOT NULL, screen_depth_m REAL NOT NULL,
 status TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hydro_endmembers (
 id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, isotope_d18o REAL NOT NULL,
 isotope_d2h REAL NOT NULL, solute_mg_l REAL NOT NULL, uncertainty REAL NOT NULL,
 sigma_d18o REAL, sigma_d2h REAL, sigma_solute REAL, covariance_json TEXT NOT NULL DEFAULT '',
 version TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)), created_at TEXT NOT NULL,
 UNIQUE(name,version)
);
CREATE TABLE IF NOT EXISTS hydro_samples (
 id INTEGER PRIMARY KEY AUTOINCREMENT, well_id INTEGER NOT NULL REFERENCES hydro_wells(id) ON DELETE RESTRICT,
 sample_code TEXT NOT NULL UNIQUE, sampled_at TEXT NOT NULL, isotope_d18o REAL, isotope_d2h REAL,
 solute_mg_l REAL, detection_limit REAL NOT NULL, measurement_error REAL NOT NULL,
 sigma_d18o REAL, sigma_d2h REAL, sigma_solute REAL, covariance_json TEXT NOT NULL DEFAULT '',
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
 status TEXT NOT NULL DEFAULT 'queued', result_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hydro_audit (
 id INTEGER PRIMARY KEY AUTOINCREMENT, resource_type TEXT NOT NULL, resource_id INTEGER,
 action TEXT NOT NULL, actor TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hydro_samples_well ON hydro_samples(well_id,sampled_at);
CREATE INDEX IF NOT EXISTS idx_hydro_inversions_status ON hydro_inversions(status,created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_schema() -> None:
    connection = get_connection()
    connection.executescript(SCHEMA)
    # 兼容已存在的旧库：补齐逐指标不确定度与协方差列
    existing = {row["name"] for row in connection.execute("PRAGMA table_info(hydro_samples)").fetchall()}
    for column, declaration in (
        ("sigma_d18o", "REAL"),
        ("sigma_d2h", "REAL"),
        ("sigma_solute", "REAL"),
        ("covariance_json", "TEXT NOT NULL DEFAULT ''"),
    ):
        if column not in existing:
            connection.execute(f"ALTER TABLE hydro_samples ADD COLUMN {column} {declaration}")
    existing = {row["name"] for row in connection.execute("PRAGMA table_info(hydro_endmembers)").fetchall()}
    for column, declaration in (
        ("sigma_d18o", "REAL"),
        ("sigma_d2h", "REAL"),
        ("sigma_solute", "REAL"),
        ("covariance_json", "TEXT NOT NULL DEFAULT ''"),
    ):
        if column not in existing:
            connection.execute(f"ALTER TABLE hydro_endmembers ADD COLUMN {column} {declaration}")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


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
        covariance_json=json.dumps(payload["covariance"],ensure_ascii=False) if payload.get("covariance") is not None else ""
        with transaction(immediate=True) as connection:
            cursor=connection.execute(
                "INSERT INTO hydro_endmembers(name,isotope_d18o,isotope_d2h,solute_mg_l,uncertainty,sigma_d18o,sigma_d2h,sigma_solute,covariance_json,version,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (payload["name"],payload["isotope_d18o"],payload["isotope_d2h"],payload["solute_mg_l"],
                 payload["uncertainty"],payload.get("sigma_d18o"),payload.get("sigma_d2h"),payload.get("sigma_solute"),
                 covariance_json,payload["version"],now))
            return dict(connection.execute("SELECT * FROM hydro_endmembers WHERE id=?",(cursor.lastrowid,)).fetchone())

    def add_sample(self, well_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        if self.connection.execute("SELECT id FROM hydro_wells WHERE id=?",(well_id,)).fetchone() is None: raise KeyError("well_not_found")
        values=[payload.get("isotope_d18o"),payload.get("isotope_d2h"),payload.get("solute_mg_l")]
        quality="usable" if sum(v is not None for v in values)>=2 else "incomplete"
        now=_now()
        covariance_json=json.dumps(payload["covariance"],ensure_ascii=False) if payload.get("covariance") is not None else ""
        with transaction(immediate=True) as connection:
            cursor=connection.execute(
                "INSERT INTO hydro_samples(well_id,sample_code,sampled_at,isotope_d18o,isotope_d2h,solute_mg_l,"
                "detection_limit,measurement_error,sigma_d18o,sigma_d2h,sigma_solute,covariance_json,quality_status,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (well_id,payload["sample_code"],payload["sampled_at"],payload.get("isotope_d18o"),payload.get("isotope_d2h"),
                 payload.get("solute_mg_l"),payload["detection_limit"],payload["measurement_error"],
                 payload.get("sigma_d18o"),payload.get("sigma_d2h"),payload.get("sigma_solute"),covariance_json,quality,now))
            return dict(connection.execute("SELECT * FROM hydro_samples WHERE id=?",(cursor.lastrowid,)).fetchone())

    @staticmethod
    def _parse_covariance(raw: str) -> list[list[float]] | None:
        if not raw:
            return None
        try:
            value=json.loads(raw)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value,list) and len(value)==3 else None

    def _resolve_weights(self, sample: sqlite3.Row, endmembers: list[sqlite3.Row], payload: dict[str, Any]) -> dict[str, Any]:
        """合并档案不确定度与请求级覆盖，生成确定性权重快照。

        优先级：请求级覆盖 > 样本/端元逐指标标准差 > 统一 measurement_error/uncertainty。
        返回的快照同时记录每个数值的来源，便于科研人员复算。
        """
        tracer_sigma_columns={
            "isotope_d18o":"sigma_d18o",
            "isotope_d2h":"sigma_d2h",
            "solute_mg_l":"sigma_solute",
        }
        obs_override=payload.get("observation_weights") or {}
        end_override_rows={
            int(row["endmember_id"]): row for row in (payload.get("endmember_weights") or [])
        }

        observation_sigmas: dict[str,float]={}
        observation_sources: dict[str,str]={}
        for tracer,column in tracer_sigma_columns.items():
            override_key={"isotope_d18o":"sigma_d18o","isotope_d2h":"sigma_d2h","solute_mg_l":"sigma_solute"}[tracer]
            if obs_override.get(override_key) is not None:
                observation_sigmas[tracer]=float(obs_override[override_key])
                observation_sources[tracer]="request_override"
            elif sample[column] is not None:
                observation_sigmas[tracer]=float(sample[column])
                observation_sources[tracer]="sample_record"
            else:
                observation_sigmas[tracer]=float(sample["measurement_error"])
                observation_sources[tracer]="measurement_error_fallback"
        if obs_override.get("covariance") is not None:
            observation_covariance=obs_override["covariance"]
            obs_cov_source="request_override"
        else:
            observation_covariance=self._parse_covariance(sample["covariance_json"])
            obs_cov_source="sample_record" if observation_covariance is not None else "none"

        solver_endmembers=[]
        endmember_weights: dict[str,dict[str,Any]]={}
        for endmember in endmembers:
            override=end_override_rows.get(endmember["id"],{})
            sigmas: dict[str,float]={}
            sources: dict[str,str]={}
            for tracer,column in tracer_sigma_columns.items():
                override_key={"isotope_d18o":"sigma_d18o","isotope_d2h":"sigma_d2h","solute_mg_l":"sigma_solute"}[tracer]
                if override.get(override_key) is not None:
                    sigmas[tracer]=float(override[override_key])
                    sources[tracer]="request_override"
                elif endmember[column] is not None:
                    sigmas[tracer]=float(endmember[column])
                    sources[tracer]="endmember_record"
                else:
                    sigmas[tracer]=float(endmember["uncertainty"])
                    sources[tracer]="uncertainty_fallback"
            if override.get("covariance") is not None:
                covariance=override["covariance"]
                cov_source="request_override"
            else:
                covariance=self._parse_covariance(endmember["covariance_json"])
                cov_source="endmember_record" if covariance is not None else "none"
            solver_endmembers.append({
                "id":endmember["id"],
                "name":endmember["name"],
                "signatures":{tracer:float(endmember[tracer]) for tracer in TRACERS},
                "sigmas":sigmas,
                "covariance":covariance,
            })
            endmember_weights[str(endmember["id"])]={
                "name":endmember["name"],
                "sigmas":sigmas,
                "sigma_sources":sources,
                "covariance":covariance,
                "covariance_source":cov_source,
            }

        return {
            "observation_sigmas":observation_sigmas,
            "observation_covariance":observation_covariance,
            "observation_sigma_sources":observation_sources,
            "observation_covariance_source":obs_cov_source,
            "endmembers":solver_endmembers,
            "endmember_weights":endmember_weights,
            "solver_version":SOLVER_VERSION,
        }

    def get_inversion(self, task_id: int) -> dict[str, Any] | None:
        task=self.connection.execute("SELECT * FROM hydro_inversions WHERE id=?",(task_id,)).fetchone()
        return dict(task) if task is not None else None

    def solve_mixture(self, sample_data: dict[str, Any], endmembers: list[dict[str, Any]], weights: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        observed=[sample_data.get(tracer) for tracer in TRACERS]
        return solve_weighted_mixture(
            observed,
            endmembers,
            weights,
            max_iterations=options["max_iterations"],
            tolerance=options["tolerance"],
            method=options["method"],
            model_version=options["model_version"],
        )

    def enqueue_inversion(self, sample_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        sample=self.connection.execute("SELECT * FROM hydro_samples WHERE id=?",(sample_id,)).fetchone()
        if sample is None: raise KeyError("sample_not_found")
        ids=sorted(set(payload["endmember_ids"]))
        endmembers=self.connection.execute(f"SELECT * FROM hydro_endmembers WHERE active=1 AND id IN ({','.join('?' for _ in ids)}) ORDER BY id",ids).fetchall()
        if len(endmembers)!=len(ids): raise ValueError("endmember_not_found")
        sample_dict=dict(sample)
        weights=self._resolve_weights(sample,endmembers,payload)
        observed=[sample_dict.get(tracer) for tracer in TRACERS]
        if sum(v is not None for v in observed)<2: raise ValueError("insufficient_measurements")
        # 确定性任务快照：原始观测、解析后权重与求解器/模型摘要
        input_data={
            "request":{
                "method":payload["method"],
                "max_iterations":payload["max_iterations"],
                "tolerance":payload["tolerance"],
                "model_version":payload["model_version"],
                "observation_weights":payload.get("observation_weights"),
                "endmember_weights":payload.get("endmember_weights"),
            },
            "endmember_ids":ids,
            "sample":sample_dict,
            "endmembers":[dict(e) for e in endmembers],
            "resolved_weights":weights,
            "model_summary":{
                "solver_version":SOLVER_VERSION,
                "model_version":payload["model_version"],
                "method":payload["method"],
            },
        }
        key=_digest(input_data); now=_now()
        with transaction(immediate=True) as connection:
            old=connection.execute("SELECT * FROM hydro_inversions WHERE task_key=?",(key,)).fetchone()
            if old: return dict(old)
            cursor=connection.execute("INSERT INTO hydro_inversions(sample_id,task_key,model_version,method,input_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",(sample_id,key,payload["model_version"],payload["method"],json.dumps(input_data,ensure_ascii=False),now,now))
            return dict(connection.execute("SELECT * FROM hydro_inversions WHERE id=?",(cursor.lastrowid,)).fetchone())

    def run_inversion(self, task_id: int, worker_id: str) -> dict[str, Any]:
        with transaction(immediate=True) as connection:
            task=connection.execute("SELECT * FROM hydro_inversions WHERE id=?",(task_id,)).fetchone()
            if task is None: raise KeyError("task_not_found")
            if task["status"]=="done": return dict(task)
            connection.execute("UPDATE hydro_inversions SET status='running',attempts=attempts+1,worker_id=?,updated_at=? WHERE id=?",(worker_id,_now(),task_id))
        data=json.loads(task["input_json"])
        request=data["request"]
        weights=data["resolved_weights"]
        try:
            result=self.solve_mixture(
                data["sample"],
                weights["endmembers"],
                weights,
                request,
            )
        except MixingError as exc:
            with transaction(immediate=True) as connection: connection.execute("UPDATE hydro_inversions SET status='failed',error=?,updated_at=? WHERE id=?",(str(exc),_now(),task_id))
            raise ValueError(str(exc)) from exc
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
