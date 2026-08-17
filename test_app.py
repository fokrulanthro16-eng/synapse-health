"""
Automated Pytest Test Suite for SynapseHealth Clinical Triage Engine
"""

import pytest
from fastapi.testclient import TestClient
from app import (
    app, VitalSigns, PatientIntake, calculate_news2, evaluate_esi_level,
    TriageAcuityAgent, DiagnosticAgent, ClinicalPharmacistAgent, VisionDiagnosticAgent,
    MultiAgentOrchestrator, TelemetryStore, OverrideRequest
)

client = TestClient(app)

# ==========================================
# 1. UNIT TESTS: CLINICAL SCORING ALGORITHMS
# ==========================================

def test_news2_calculation_normal():
    vitals = VitalSigns(
        heart_rate=72, bp_systolic=120, bp_diastolic=80,
        spo2=98.0, temperature_c=36.8, respiratory_rate=16,
        gcs=15, on_supplemental_o2=False
    )
    score = calculate_news2(vitals)
    assert score == 0

def test_news2_calculation_high_risk():
    vitals = VitalSigns(
        heart_rate=135, bp_systolic=85, bp_diastolic=50,
        spo2=89.0, temperature_c=39.5, respiratory_rate=28,
        gcs=13, on_supplemental_o2=True
    )
    score = calculate_news2(vitals)
    assert score >= 10  # Critical NEWS2 score

def test_esi_evaluation_level_1():
    patient = PatientIntake(
        name="Unresponsive Patient", age=65, gender="Male",
        chief_complaint="Unresponsive, severe respiratory distress",
        symptoms=["Unresponsive", "Hypoxia"],
        vitals=VitalSigns(
            heart_rate=145, bp_systolic=75, bp_diastolic=45,
            spo2=82.0, temperature_c=36.0, respiratory_rate=34,
            gcs=7, on_supplemental_o2=True
        )
    )
    esi = evaluate_esi_level(patient, news2_score=12)
    assert esi["esi_level"] == 1
    assert esi["acuity_label"] == "Resuscitation"
    assert esi["life_threat"] is True

def test_esi_evaluation_level_2():
    patient = PatientIntake(
        name="Chest Pain Patient", age=55, gender="Female",
        chief_complaint="Crushing retrosternal chest pain",
        symptoms=["Chest pain", "Diaphoresis"],
        vitals=VitalSigns(
            heart_rate=110, bp_systolic=150, bp_diastolic=95,
            spo2=95.0, temperature_c=37.0, respiratory_rate=22,
            gcs=15, on_supplemental_o2=False
        )
    )
    esi = evaluate_esi_level(patient, news2_score=3)
    assert esi["esi_level"] == 2
    assert esi["acuity_label"] == "Emergent"

# ==========================================
# 2. UNIT TESTS: MICRO-AGENTS & VISION ENGINE
# ==========================================

@pytest.mark.asyncio
async def test_triage_acuity_agent():
    patient = PatientIntake(
        name="Test Patient", age=45, gender="Male",
        chief_complaint="Shortness of breath",
        symptoms=["Dyspnea"],
        vitals=VitalSigns(
            heart_rate=105, bp_systolic=135, bp_diastolic=85,
            spo2=94.0, temperature_c=37.2, respiratory_rate=22,
            gcs=15, on_supplemental_o2=False
        )
    )
    res = await TriageAcuityAgent.analyze(patient)
    assert res.agent_name == "Triage Acuity Agent"
    assert "esi_level" in res.details
    assert res.confidence >= 0.8

@pytest.mark.asyncio
async def test_diagnostic_agent_stemi():
    patient = PatientIntake(
        name="STEMI Test", age=60, gender="Male",
        chief_complaint="Severe retrosternal pain radiating to left arm",
        symptoms=["Chest pain", "Diaphoresis"],
        vitals=VitalSigns(
            heart_rate=115, bp_systolic=160, bp_diastolic=95,
            spo2=94.0, temperature_c=36.8, respiratory_rate=22,
            gcs=15, on_supplemental_o2=False
        )
    )
    res = await DiagnosticAgent.analyze(patient, {"esi_level": 2})
    assert res.agent_name == "Diagnostic Reasoning Agent"
    assert len(res.details["differentials"]) > 0
    top_cond = res.details["differentials"][0]["condition"]
    assert "Coronary Syndrome" in top_cond or "STEMI" in top_cond

@pytest.mark.asyncio
async def test_clinical_pharmacist_agent_allergy():
    patient = PatientIntake(
        name="Allergy Test", age=30, gender="Female",
        chief_complaint="Fever and dysuria",
        symptoms=["Fever"],
        vitals=VitalSigns(
            heart_rate=90, bp_systolic=120, bp_diastolic=80,
            spo2=98.0, temperature_c=38.5, respiratory_rate=18,
            gcs=15, on_supplemental_o2=False
        ),
        allergies=["Penicillin"],
        current_medications=["Metformin 500mg"],
        medical_history=["Chronic Kidney Disease"]
    )
    diag_details = {"differentials": [{"condition": "Severe Sepsis", "icd10": "A41.9", "probability": 80.0, "red_flag": True}]}
    res = await ClinicalPharmacistAgent.analyze(patient, diag_details)
    assert res.agent_name == "Clinical Pharmacist Agent"
    alerts = res.details["alerts"]
    assert any("Penicillin" in a["medication"] for a in alerts)
    assert any("Metformin" in a["medication"] for a in alerts)

@pytest.mark.asyncio
async def test_vision_diagnostic_agent():
    # Simple 10x10 dummy base64 pixel image
    dummy_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSU5GGhAAAABJREFUeJzs0SERgDAUA8E9B6SgBAUoSgISUJA0M2H3s0x3v7v+c7m7y+3uLrf7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7v7vwB4xgEN7GgqEwAAAABJRU5ErkJggg=="
    res = await VisionDiagnosticAgent.analyze_image_features(dummy_b64, "Erythematous skin rash")
    assert res.agent_name == "Vision Diagnostic Agent"
    assert "visual_findings" in res.details
    assert res.confidence > 0.5

# ==========================================
# 3. END-TO-END FASTAPI INTEGRATION TESTS
# ==========================================

def test_api_get_dashboard():
    response = client.get("/")
    assert response.status_code == 200
    assert "SynapseHealth" in response.text

def test_api_get_presets():
    response = client.get("/api/patients/sample")
    assert response.status_code == 200
    data = response.json()
    assert "acute_coronary_syndrome" in data
    assert "severe_sepsis" in data

def test_api_get_metrics():
    response = client.get("/api/system/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_patients_triaged" in data
    assert "agent_status" in data

def test_api_post_triage_analyze():
    payload = {
        "name": "Integration Test Patient",
        "age": 50,
        "gender": "Male",
        "chief_complaint": "Acute severe headache",
        "symptoms": ["Headache", "Photophobia"],
        "vitals": {
            "heart_rate": 88,
            "bp_systolic": 130,
            "bp_diastolic": 85,
            "spo2": 98.0,
            "temperature_c": 37.0,
            "respiratory_rate": 16,
            "gcs": 15,
            "on_supplemental_o2": False
        },
        "medical_history": [],
        "current_medications": [],
        "allergies": []
    }
    response = client.post("/api/triage/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "esi_level" in data
    assert "acuity_label" in data
    assert data["triage_agent"]["agent_name"] == "Triage Acuity Agent"

def test_api_post_override():
    override_payload = {
        "patient_id": "PT-TEST-001",
        "overridden_esi": 1,
        "clinician_id": "DR-PYTEST",
        "notes": "Automated integration test override"
    }
    response = client.post("/api/agents/override", json=override_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
