from __future__ import annotations

import json


def create_well(client,code="W-001"):
    response=client.post("/api/hydro/wells",json={"code":code,"name":"北部监测井","latitude":35.1,"longitude":116.2,"aquifer":"浅层孔隙含水层","screen_depth_m":42})
    assert response.status_code==201,response.text
    return response.json()


def test_mixture_inversion_and_transport(client):
    well=create_well(client)
    e1=client.post("/api/hydro/endmembers",json={"name":"山区降水","isotope_d18o":-10,"isotope_d2h":-70,"solute_mg_l":10,"uncertainty":0.1,"version":"v1"}).json()
    e2=client.post("/api/hydro/endmembers",json={"name":"河流渗漏","isotope_d18o":-5,"isotope_d2h":-35,"solute_mg_l":50,"uncertainty":0.2,"version":"v1"}).json()
    sample=client.post(f"/api/hydro/wells/{well['id']}/samples",json={"sample_code":"S-001","sampled_at":"2026-09-24T08:00:00+00:00","isotope_d18o":-7.5,"isotope_d2h":-52.5,"solute_mg_l":30,"detection_limit":0.1,"measurement_error":0.05}).json()
    task=client.post(f"/api/hydro/samples/{sample['id']}/inversions",json={"endmember_ids":[e1['id'],e2['id']],"max_iterations":1000,"tolerance":1e-10,"model_version":"mix-test"})
    assert task.status_code==202,task.text
    done=client.post(f"/api/hydro/inversions/{task.json()['id']}/run?worker_id=test")
    assert done.status_code==200,done.text
    assert done.json()["status"]=="done"
    transport=client.post(f"/api/hydro/wells/{well['id']}/transport",json={"source_concentration":100,"distance_m":100,"velocity_m_day":2,"dispersion_m2_day":5,"decay_per_day":0.01,"duration_days":100,"step_days":5,"model_version":"ade-test"})
    assert transport.status_code==201,transport.text
    assert transport.json()["result_json"]


def test_missing_measurement_is_classified(client):
    well=create_well(client,"W-002")
    sample=client.post(f"/api/hydro/wells/{well['id']}/samples",json={"sample_code":"S-002","sampled_at":"2026-09-24T08:00:00+00:00","isotope_d18o":-7.5,"detection_limit":0.1,"measurement_error":0.05})
    assert sample.status_code==201
    assert sample.json()["quality_status"]=="incomplete"


def _setup_mixing_case(client,code="W-100"):
    well=create_well(client,code)
    rain=client.post("/api/hydro/endmembers",json={
        "name":"山区降水","isotope_d18o":-10,"isotope_d2h":-70,"solute_mg_l":10,
        "uncertainty":0.1,"sigma_d18o":0.2,"sigma_d2h":1.0,"sigma_solute":2.0,"version":"v2",
    }).json()
    river=client.post("/api/hydro/endmembers",json={
        "name":"河流渗漏","isotope_d18o":-5,"isotope_d2h":-35,"solute_mg_l":50,
        "uncertainty":0.2,"sigma_d18o":0.15,"sigma_d2h":0.8,"sigma_solute":3.0,"version":"v2",
    }).json()
    deep=client.post("/api/hydro/endmembers",json={
        "name":"深层地下水","isotope_d18o":-8,"isotope_d2h":-60,"solute_mg_l":200,
        "uncertainty":0.3,"sigma_d18o":0.1,"sigma_d2h":0.5,"sigma_solute":5.0,"version":"v2",
    }).json()
    sample=client.post(f"/api/hydro/wells/{well['id']}/samples",json={
        "sample_code":"S-100","sampled_at":"2026-09-24T08:00:00+00:00",
        "isotope_d18o":-7.0,"isotope_d2h":-46.0,"solute_mg_l":88.0,
        "detection_limit":0.1,"measurement_error":0.05,
        "sigma_d18o":0.1,"sigma_d2h":0.5,"sigma_solute":2.0,
    }).json()
    return well,rain,river,deep,sample


def test_weighted_inversion_reports_constraints_residuals_and_identifiability(client):
    well,rain,river,deep,sample=_setup_mixing_case(client)
    payload={
        "endmember_ids":[rain["id"],river["id"],deep["id"]],
        "max_iterations":1000,"tolerance":1e-10,"model_version":"mix-wls-test",
        "observation_weights":{
            "sigma_d18o":0.08,"sigma_d2h":0.4,"sigma_solute":1.5,
            "covariance":[[0.0064,0.01,0.0],[0.01,0.16,0.0],[0.0,0.0,2.25]],
        },
    }
    task=client.post(f"/api/hydro/samples/{sample['id']}/inversions",json=payload)
    assert task.status_code==202,task.text
    done=client.post(f"/api/hydro/inversions/{task.json()['id']}/run?worker_id=test")
    assert done.status_code==200,done.text
    body=done.json()
    assert body["status"]=="done"
    result=json.loads(body["result_json"])
    fractions=result["fractions"]
    # 比例非负、总和为一
    assert all(row["fraction"]>=0.0 for row in fractions)
    assert abs(result["mass_balance"]-1.0)<1e-8
    # 每个端元都有确定性名次，比例行按端元 id 升序输出
    assert [row["endmember_id"] for row in fractions]==sorted([rain["id"],river["id"],deep["id"]])
    assert all(row["rank"]>=1 for row in fractions)
    # 加权残差、收敛原因、可辨识性
    assert result["converged"] is True
    assert result["convergence_reason"]=="converged"
    assert result["identifiability"]["status"]=="identified"
    assert len(result["residuals"])==3
    assert all("standardized_residual" in row for row in result["residuals"])
    assert "weighted_residuals" in result and len(result["weighted_residuals"])==3
    assert result["chi_square"]>=0.0
    assert result["model_summary"]["solver_version"]
    assert result["model_summary"]["constraints"]==[
        "fractions_non_negative","fractions_sum_to_one","conservative_mass_balance"]


def test_same_inputs_produce_same_task_and_deterministic_order(client):
    well,rain,river,deep,sample=_setup_mixing_case(client,"W-101")
    payload={"endmember_ids":[deep["id"],rain["id"],river["id"]],"model_version":"mix-det"}
    first=client.post(f"/api/hydro/samples/{sample['id']}/inversions",json=payload)
    assert first.status_code==202
    second=client.post(f"/api/hydro/samples/{sample['id']}/inversions",json=payload)
    assert second.status_code==202
    # 相同输入与模型版本 → 同一任务
    assert first.json()["id"]==second.json()["id"]
    assert first.json()["task_key"]==second.json()["task_key"]
    done=client.post(f"/api/hydro/inversions/{first.json()['id']}/run?worker_id=w1").json()
    result=json.loads(done["result_json"])
    run_again=client.post(f"/api/hydro/inversions/{first.json()['id']}/run?worker_id=w2").json()
    # 已完成任务重复执行返回同一结果
    assert json.loads(run_again["result_json"])==result
    fractions=result["fractions"]
    # 比例按端元 id 升序输出，名次由比例决定，并列时同名次
    assert [row["endmember_id"] for row in fractions]==sorted([rain["id"],river["id"],deep["id"]])
    values=[row["fraction"] for row in fractions]
    by_rank=sorted(fractions,key=lambda row:(row["rank"],row["endmember_id"]))
    assert by_rank[0]["fraction"]==max(values)
    # 名次与比例降序一致
    ranked_values=[row["fraction"] for row in by_rank]
    assert ranked_values==sorted(values,reverse=True)


def test_task_snapshot_preserves_observations_weights_and_model_summary(client):
    well,rain,river,deep,sample=_setup_mixing_case(client,"W-102")
    payload={
        "endmember_ids":[rain["id"],deep["id"]],
        "model_version":"mix-snap",
        "endmember_weights":[
            {"endmember_id":rain["id"],"sigma_solute":4.0},
        ],
    }
    task=client.post(f"/api/hydro/samples/{sample['id']}/inversions",json=payload).json()
    stored=client.get(f"/api/hydro/inversions/{task['id']}")
    assert stored.status_code==200
    snapshot=json.loads(stored.json()["input_json"])
    # 原始观测保留
    assert snapshot["sample"]["sample_code"]=="S-100"
    assert snapshot["sample"]["isotope_d18o"]==-7.0
    # 解析后的权重与来源保留
    weights=snapshot["resolved_weights"]
    assert weights["observation_sigmas"]["solute_mg_l"]==2.0
    assert weights["observation_sigma_sources"]["isotope_d18o"]=="sample_record"
    assert str(rain["id"]) in weights["endmember_weights"]
    assert weights["endmember_weights"][str(rain["id"])]["sigmas"]["solute_mg_l"]==4.0
    assert weights["endmember_weights"][str(rain["id"])]["sigma_sources"]["solute_mg_l"]=="request_override"
    assert weights["solver_version"]
    # 模型摘要保留
    assert snapshot["model_summary"]["model_version"]=="mix-snap"
    assert snapshot["model_summary"]["solver_version"]==weights["solver_version"]


def test_high_uncertainty_tracer_does_not_dominate(client):
    well,rain,river,deep,sample=_setup_mixing_case(client,"W-103")
    # 溶质观测偏向深层端元（200 mg/L 端），但赋予极大不确定度；
    # 高精度同位素指标应主导，使深层占比不致被溶质单点拉满。
    payload={
        "endmember_ids":[rain["id"],river["id"],deep["id"]],
        "model_version":"mix-weights",
        "observation_weights":{"sigma_solute":5000.0},
    }
    task=client.post(f"/api/hydro/samples/{sample['id']}/inversions",json=payload).json()
    result=json.loads(client.post(f"/api/hydro/inversions/{task['id']}/run?worker_id=w").json()["result_json"])
    fractions={row["endmember_id"]:row["fraction"] for row in result["fractions"]}
    assert result["convergence_reason"]=="converged"
    assert fractions[deep["id"]]<0.9


def test_unidentifiable_configuration_is_flagged(client):
    well=create_well(client,"W-104")
    e1=client.post("/api/hydro/endmembers",json={"name":"降水A","isotope_d18o":-10,"isotope_d2h":-70,"solute_mg_l":10,"uncertainty":0.1,"version":"v9"}).json()
    e2=client.post("/api/hydro/endmembers",json={"name":"河水B","isotope_d18o":-5,"isotope_d2h":-35,"solute_mg_l":50,"uncertainty":0.2,"version":"v9"}).json()
    e3=client.post("/api/hydro/endmembers",json={"name":"深层C","isotope_d18o":-8,"isotope_d2h":-55,"solute_mg_l":30,"uncertainty":0.2,"version":"v9"}).json()
    # 只有两个指标却有三个端元
    sample=client.post(f"/api/hydro/wells/{well['id']}/samples",json={
        "sample_code":"S-104","sampled_at":"2026-09-24T08:00:00+00:00",
        "isotope_d18o":-7.5,"isotope_d2h":-52.5,
        "detection_limit":0.1,"measurement_error":0.05,
    }).json()
    task=client.post(f"/api/hydro/samples/{sample['id']}/inversions",
                     json={"endmember_ids":[e1["id"],e2["id"],e3["id"]],"model_version":"mix-unid"}).json()
    result=json.loads(client.post(f"/api/hydro/inversions/{task['id']}/run?worker_id=w").json()["result_json"])
    assert result["identifiability"]["status"]=="unidentifiable"
    codes={warning["code"] for warning in result["identifiability"]["warnings"]}
    assert "observation_count_below_endmember_count" in codes
    # 即便不可辨识，约束仍然满足
    assert all(row["fraction"]>=0.0 for row in result["fractions"])
    assert abs(result["mass_balance"]-1.0)<1e-8


def test_invalid_covariance_is_rejected(client):
    well=create_well(client,"W-105")
    bad=client.post("/api/hydro/endmembers",json={
        "name":"坏协方差端元","isotope_d18o":-9,"isotope_d2h":-60,"solute_mg_l":12,
        "uncertainty":0.1,"covariance":[[1.0,5.0,0.0],[5.0,1.0,0.0],[0.0,0.0,1.0]],
    })
    assert bad.status_code==422,bad.text


def test_missing_task_returns_404(client):
    assert client.get("/api/hydro/inversions/99999").status_code==404
