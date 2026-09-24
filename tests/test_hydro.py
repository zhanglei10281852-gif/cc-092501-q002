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
