from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.database import get_connection, transaction


SCHEMA = """
CREATE TABLE IF NOT EXISTS hydro_wells (
 id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
 latitude REAL NOT NULL, longitude REAL NOT NULL, aquifer TEXT NOT NULL, screen_depth_m REAL NOT NULL,
 status TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hydro_endmembers (
 id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, isotope_d18o REAL NOT NULL,
 isotope_d2h REAL NOT NULL, solute_mg_l REAL NOT NULL, uncertainty REAL NOT NULL,
 version TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)), created_at TEXT NOT NULL,
 UNIQUE(name,version)
);
CREATE TABLE IF NOT EXISTS hydro_samples (
 id INTEGER PRIMARY KEY AUTOINCREMENT, well_id INTEGER NOT NULL REFERENCES hydro_wells(id) ON DELETE RESTRICT,
 sample_code TEXT NOT NULL UNIQUE, sampled_at TEXT NOT NULL, isotope_d18o REAL, isotope_d2h REAL,
 solute_mg_l REAL, detection_limit REAL NOT NULL, measurement_error REAL NOT NULL,
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
    get_connection().executescript(SCHEMA)


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
        with transaction(immediate=True) as connection:
            cursor=connection.execute("INSERT INTO hydro_endmembers(name,isotope_d18o,isotope_d2h,solute_mg_l,uncertainty,version,created_at) VALUES(?,?,?,?,?,?,?)",(payload["name"],payload["isotope_d18o"],payload["isotope_d2h"],payload["solute_mg_l"],payload["uncertainty"],payload["version"],now))
            return dict(connection.execute("SELECT * FROM hydro_endmembers WHERE id=?",(cursor.lastrowid,)).fetchone())

    def add_sample(self, well_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        if self.connection.execute("SELECT id FROM hydro_wells WHERE id=?",(well_id,)).fetchone() is None: raise KeyError("well_not_found")
        values=[payload.get("isotope_d18o"),payload.get("isotope_d2h"),payload.get("solute_mg_l")]
        quality="usable" if sum(v is not None for v in values)>=2 else "incomplete"
        now=_now()
        with transaction(immediate=True) as connection:
            cursor=connection.execute("INSERT INTO hydro_samples(well_id,sample_code,sampled_at,isotope_d18o,isotope_d2h,solute_mg_l,detection_limit,measurement_error,quality_status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",(well_id,payload["sample_code"],payload["sampled_at"],payload.get("isotope_d18o"),payload.get("isotope_d2h"),payload.get("solute_mg_l"),payload["detection_limit"],payload["measurement_error"],quality,now))
            return dict(connection.execute("SELECT * FROM hydro_samples WHERE id=?",(cursor.lastrowid,)).fetchone())

    def _project_simplex(self, values: list[float]) -> list[float]:
        clipped=[max(0.0,v) for v in values]
        total=sum(clipped)
        return [1/len(values)]*len(values) if total<=1e-15 else [v/total for v in clipped]

    def solve_mixture(self, sample: sqlite3.Row, endmembers: list[sqlite3.Row], max_iterations: int, tolerance: float) -> dict[str, Any]:
        observed=[sample["isotope_d18o"],sample["isotope_d2h"],sample["solute_mg_l"]]
        active=[i for i,v in enumerate(observed) if v is not None]
        if len(active)<2: raise ValueError("insufficient_measurements")
        fractions=[1/len(endmembers)]*len(endmembers)
        scale=[20.0,100.0,max(1.0,float(sample["solute_mg_l"] or 1))]
        rate=0.08
        last=float("inf")
        for iteration in range(max_iterations):
            predicted=[sum(fractions[j]*[e["isotope_d18o"],e["isotope_d2h"],e["solute_mg_l"]][k] for j,e in enumerate(endmembers)) for k in range(3)]
            residual=[(predicted[k]-float(observed[k]))/scale[k] if k in active else 0.0 for k in range(3)]
            objective=sum(r*r for r in residual)+((sum(fractions)-1.0)*10)**2
            if abs(last-objective)<tolerance: break
            last=objective
            gradient=[]
            for e in endmembers:
                vector=[e["isotope_d18o"],e["isotope_d2h"],e["solute_mg_l"]]
                gradient.append(2*sum(residual[k]*vector[k]/scale[k] for k in active))
            fractions=self._project_simplex([f-rate*g for f,g in zip(fractions,gradient)])
        predicted=[sum(fractions[j]*[e["isotope_d18o"],e["isotope_d2h"],e["solute_mg_l"]][k] for j,e in enumerate(endmembers)) for k in range(3)]
        rmse=math.sqrt(sum(((predicted[k]-float(observed[k]))/scale[k])**2 for k in active)/len(active))
        return {"fractions":[{"endmember_id":e["id"],"name":e["name"],"fraction":round(f,8)} for e,f in zip(endmembers,fractions)],"mass_balance":round(sum(fractions),10),"predicted":predicted,"rmse":rmse,"iterations":iteration+1,"converged":abs(last-objective)<tolerance}

    def enqueue_inversion(self, sample_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        sample=self.connection.execute("SELECT * FROM hydro_samples WHERE id=?",(sample_id,)).fetchone()
        if sample is None: raise KeyError("sample_not_found")
        ids=sorted(set(payload["endmember_ids"]))
        endmembers=self.connection.execute(f"SELECT * FROM hydro_endmembers WHERE active=1 AND id IN ({','.join('?' for _ in ids)}) ORDER BY id",ids).fetchall()
        if len(endmembers)!=len(ids): raise ValueError("endmember_not_found")
        input_data={**payload,"endmember_ids":ids,"sample":dict(sample),"endmembers":[dict(e) for e in endmembers]}
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
        sample=self.connection.execute("SELECT * FROM hydro_samples WHERE id=?",(task["sample_id"],)).fetchone()
        ids=data["endmember_ids"]
        endmembers=self.connection.execute(f"SELECT * FROM hydro_endmembers WHERE id IN ({','.join('?' for _ in ids)}) ORDER BY id",ids).fetchall()
        try: result=self.solve_mixture(sample,endmembers,data["max_iterations"],data["tolerance"])
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
