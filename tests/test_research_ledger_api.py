from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from eti.api import app


def test_authenticated_trial_ledger_drives_statistical_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "eti.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    with TestClient(app) as client:
        account = client.post("/v1/auth/register", json={
            "username":"ledger_owner","password":"a-long-password-123",
            "invite_code":"private-invite-code-123456"})
        assert account.status_code == 201, account.text
        headers={"Authorization":f"Bearer {account.json()['access_token']}"}
        workspace=client.post("/v1/workspaces",headers=headers,json={"name":"research"}).json()
        wid=workspace["id"]
        base={"family_id":"family-1","training_data_sha256":"a"*64,
              "evaluation_data_sha256":"b"*64,"cost_model_sha256":"c"*64,
              "benchmark_id":"cash"}
        for i in range(3):
            payload={**base,"trial_id":f"trial-{i}","strategy_name":f"strategy-{i}",
                     "config_sha256":str(i+1)*64}
            response=client.post(f"/v1/workspaces/{wid}/research-trials",
                                 headers=headers,json=payload)
            assert response.status_code==201,response.text
            assert client.post(f"/v1/workspaces/{wid}/research-trials/trial-{i}/start",
                               headers=headers).status_code==200
            if i==2:
                failed=client.post(f"/v1/workspaces/{wid}/research-trials/trial-{i}/fail",
                                   headers=headers,json={"reason":"optimizer failure"})
                assert failed.status_code==200,failed.text
                continue
            start=datetime(2026,1,1,tzinfo=timezone.utc)
            times=[(start+timedelta(hours=j)).isoformat().replace("+00:00","Z") for j in range(32)]
            values=[(0.001-i*0.0002)+((j%5)-2)*0.0001 for j in range(32)]
            complete=client.post(f"/v1/workspaces/{wid}/research-trials/trial-{i}/complete",
                headers=headers,json={"observation_times":times,"net_return_differentials":values})
            assert complete.status_code==200,complete.text
        result=client.post(f"/v1/workspaces/{wid}/research-trial-families/family-1/diagnostics",
            headers=headers,json={"n_splits":8,"repetitions":200,"block_length":3,"seed":9})
        assert result.status_code==200,result.text
        report=result.json()
        assert report["attempted_trial_count"]==3
        assert report["completed_trial_count"]==2
        assert report["failed_trial_count"]==1
        assert report["selected_trial_deflated_information_ratio"]["trial_count"]==3
        assert 0 <= report["family_pbo"]["pbo"] <= 1
        assert report["white_reality_check"]["p_value"] > 0
        assert report["hansen_spa"]["p_value"] > 0
        other=client.get(f"/v1/workspaces/{wid}/research-trial-families/family-1",
                         headers={"Authorization":"Bearer invalid"})
        assert other.status_code==401
