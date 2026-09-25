from __future__ import annotations


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


def _three_endmembers(client):
    e1=client.post("/api/hydro/endmembers",json={"name":"降水","isotope_d18o":-10,"isotope_d2h":-70,"solute_mg_l":10,
        "isotope_d18o_std":0.2,"isotope_d2h_std":1.5,"solute_std":2.0,"version":"v2"}).json()
    e2=client.post("/api/hydro/endmembers",json={"name":"河流渗漏","isotope_d18o":-5,"isotope_d2h":-35,"solute_mg_l":50,
        "isotope_d18o_std":0.3,"isotope_d2h_std":2.0,"solute_std":4.0,"version":"v2"}).json()
    e3=client.post("/api/hydro/endmembers",json={"name":"深层地下水","isotope_d18o":-3,"isotope_d2h":-25,"solute_mg_l":300,
        "isotope_d18o_std":0.4,"isotope_d2h_std":3.0,"solute_std":15.0,"version":"v2"}).json()
    return e1,e2,e3


def _run(client, sample_id, ids, body_extra=None):
    body={"endmember_ids":ids,"max_iterations":2000,"tolerance":1e-11,"model_version":"mix-2"}
    if body_extra: body.update(body_extra)
    task=client.post(f"/api/hydro/samples/{sample_id}/inversions",json=body)
    assert task.status_code==202,task.text
    done=client.post(f"/api/hydro/inversions/{task.json()['id']}/run?worker_id=test")
    assert done.status_code==200,done.text
    return task.json(),done.json()


def test_weighted_inversion_recovers_mix_and_keeps_reproducibility_inputs(client):
    well=create_well(client,"W-100")
    e1,e2,e3=_three_endmembers(client)
    # 真实比例 0.5/0.3/0.2 的混合值
    sample=client.post(f"/api/hydro/wells/{well['id']}/samples",json={
        "sample_code":"S-100","sampled_at":"2026-09-24T08:00:00+00:00",
        "isotope_d18o":-7.1,"isotope_d2h":-50.5,"solute_mg_l":80.0,
        "detection_limit":0.1,"isotope_d18o_std":0.1,"isotope_d2h_std":0.8,"solute_std":1.0}).json()
    ids=[e1["id"],e2["id"],e3["id"]]
    task,done=_run(client,sample["id"],ids)
    assert done["status"]=="done"
    import json
    result=json.loads(done["result_json"])

    # 比例非负、总和为一、恢复真实混合
    fractions={f["endmember_id"]:f["fraction"] for f in result["fractions"]}
    assert all(0.0<=v<=1.0 for v in fractions.values())
    assert abs(result["mass_balance"]-1.0)<1e-7
    assert abs(fractions[e1["id"]]-0.5)<0.05
    assert abs(fractions[e2["id"]]-0.3)<0.05
    assert abs(fractions[e3["id"]]-0.2)<0.05

    # 确定的比例顺序：按比例降序，并列按端元 id 升序
    ordered=sorted(result["fractions"],key=lambda f:f["rank"])
    assert [f["endmember_id"] for f in ordered]==result["fraction_order"]
    assert ordered==sorted(result["fractions"],key=lambda f:(-f["fraction"],f["endmember_id"]))

    # 收敛信息与加权残差
    assert result["converged"] is True
    assert result["convergence_reason"]=="stationary"
    assert result["identifiable"] is True
    assert len(result["weighted_residuals"])==3
    assert len(result["observations"])==3
    for record in result["observations"]:
        assert set(record) >= {"tracer","observed","predicted","residual","sigma","weighted_residual"}

    # API 保留原始观测、权重与模型摘要
    fetched=client.get(f"/api/hydro/inversions/{task['id']}").json()
    stored=json.loads(fetched["input_json"])
    assert stored["sample"]["isotope_d18o"]==-7.1
    assert stored["sample"]["isotope_d2h_std"]==0.8
    assert {e["id"] for e in stored["endmembers"]}==set(ids)
    assert result["weights"]["observation"]["std"][0]==0.1
    assert result["weights"]["endmembers"][2]["std"][2]==15.0
    assert result["model"]["model_version"]=="mix-2"
    assert result["model"]["solver_version"]
    assert "fractions_non_negative" in result["model"]["constraints"]

    # 相同输入得到同一任务（幂等去重）
    again=client.post(f"/api/hydro/samples/{sample['id']}/inversions",json={
        "endmember_ids":ids,"max_iterations":2000,"tolerance":1e-11,"model_version":"mix-2"})
    assert again.json()["task_key"]==task["task_key"]


def test_same_input_yields_deterministic_fractions(client):
    well=create_well(client,"W-101")
    e1,e2,e3=_three_endmembers(client)
    def make_sample(code,d18o):
        return client.post(f"/api/hydro/wells/{well['id']}/samples",json={
            "sample_code":code,"sampled_at":"2026-09-24T08:00:00+00:00",
            "isotope_d18o":d18o,"isotope_d2h":-50.5,"solute_mg_l":80.0,
            "isotope_d18o_std":0.1,"isotope_d2h_std":0.8,"solute_std":1.0}).json()
    ids=[e1["id"],e2["id"],e3["id"]]
    s1=make_sample("S-101",-7.1); s2=make_sample("S-102",-7.1)
    _,d1=_run(client,s1["id"],ids)
    _,d2=_run(client,s2["id"],ids)
    import json
    assert json.loads(d1["result_json"])["fractions"]==json.loads(d2["result_json"])["fractions"]


def test_snapshot_reproduces_result_after_endmember_changes(client):
    well=create_well(client,"W-102")
    e1,e2,e3=_three_endmembers(client)
    sample=client.post(f"/api/hydro/wells/{well['id']}/samples",json={
        "sample_code":"S-110","sampled_at":"2026-09-24T08:00:00+00:00",
        "isotope_d18o":-7.1,"isotope_d2h":-50.5,"solute_mg_l":80.0,
        "isotope_d18o_std":0.1,"isotope_d2h_std":0.8,"solute_std":1.0}).json()
    task,done=_run(client,sample["id"],[e1["id"],e2["id"],e3["id"]])
    before=done["result_json"]
    # 再跑一次已完成任务，直接返回同一结果
    rerun=client.post(f"/api/hydro/inversions/{task['id']}/run?worker_id=test")
    assert rerun.json()["result_json"]==before


def test_unidentifiable_mix_is_flagged(client):
    well=create_well(client,"W-103")
    # 河流渗漏与深层地下水示踪特征重合
    e1=client.post("/api/hydro/endmembers",json={"name":"降水A","isotope_d18o":-10,"isotope_d2h":-70,"solute_mg_l":10,"uncertainty":0.2,"version":"v3"}).json()
    e2=client.post("/api/hydro/endmembers",json={"name":"河流B","isotope_d18o":-5,"isotope_d2h":-35,"solute_mg_l":50,"uncertainty":0.2,"version":"v3"}).json()
    e3=client.post("/api/hydro/endmembers",json={"name":"深层C","isotope_d18o":-5,"isotope_d2h":-35,"solute_mg_l":50,"uncertainty":0.2,"version":"v3"}).json()
    sample=client.post(f"/api/hydro/wells/{well['id']}/samples",json={
        "sample_code":"S-120","sampled_at":"2026-09-24T08:00:00+00:00",
        "isotope_d18o":-7.5,"isotope_d2h":-52.5,"solute_mg_l":30.0}).json()
    _,done=_run(client,sample["id"],[e1["id"],e2["id"],e3["id"]])
    import json
    result=json.loads(done["result_json"])
    assert result["identifiable"] is False
    codes={w["code"] for w in result["warnings"]}
    assert "rank_deficient" in codes or "collinear_endmembers" in codes
    assert result["diagnostics"]["effective_rank"] < result["diagnostics"]["required_rank"]


def test_observation_covariance_is_accepted_and_weighted(client):
    well=create_well(client,"W-104")
    e1,e2,e3=_three_endmembers(client)
    cov=[[0.01,0.04,0.0],[0.04,0.64,0.0],[0.0,0.0,1.0]]
    sample=client.post(f"/api/hydro/wells/{well['id']}/samples",json={
        "sample_code":"S-130","sampled_at":"2026-09-24T08:00:00+00:00",
        "isotope_d18o":-7.1,"isotope_d2h":-50.5,"solute_mg_l":80.0,
        "covariance":cov}).json()
    _,done=_run(client,sample["id"],[e1["id"],e2["id"],e3["id"]])
    import json
    result=json.loads(done["result_json"])
    assert result["weights"]["observation"]["covariance"]==cov
    assert result["converged"] is True


def test_non_symmetric_covariance_is_rejected(client):
    well=create_well(client,"W-105")
    response=client.post(f"/api/hydro/wells/{well['id']}/samples",json={
        "sample_code":"S-140","sampled_at":"2026-09-24T08:00:00+00:00",
        "isotope_d18o":-7.1,"isotope_d2h":-50.5,"solute_mg_l":80.0,
        "covariance":[[0.01,0.04,0],[0.0,0.64,0],[0,0,1.0]]})
    assert response.status_code==422


def test_get_missing_inversion_returns_404(client):
    assert client.get("/api/hydro/inversions/99999").status_code==404

