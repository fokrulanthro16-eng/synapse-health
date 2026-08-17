"""
Automated Pytest Test Suite for SynapseHealth Clinical Triage Engine
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_read_dashboard():
    """Test GET / status code and HTML title presence."""
    response = client.get("/")
    assert response.status_code == 200
    assert "SynapseHealth" in response.text


def test_get_metrics():
    """Test GET /api/system/metrics returns system uptime and online agent status."""
    response = client.get("/api/system/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "system_uptime" in data
    assert "agent_status" in data
    assert data["agent_status"]["orchestrator_engine"] == "ONLINE"


def test_get_preset_samples():
    """Test GET /api/patients/sample returns clinical case scenarios."""
    response = client.get("/api/patients/sample")
    assert response.status_code == 200
    data = response.json()
    assert "acute_coronary_syndrome" in data
    assert "severe_sepsis" in data


def test_post_triage_analyze():
    """Test POST /api/triage/analyze returns acuity rating and diagnostic recommendations."""
    payload = {
        "name": "Test Patient",
        "age": 58,
        "gender": "Male",
        "chief_complaint": "Severe retrosternal chest pain radiating to left arm",
        "symptoms": ["Chest pain", "Diaphoresis", "Dyspnea"],
        "vitals": {
            "heart_rate": 118,
            "bp_systolic": 168,
            "bp_diastolic": 98,
            "spo2": 93.5,
            "temperature_c": 36.8,
            "respiratory_rate": 24,
            "gcs": 15,
            "on_supplemental_o2": False
        },
        "medical_history": ["Hypertension"],
        "current_medications": ["Lisinopril 20mg"],
        "allergies": ["Penicillin"]
    }
    response = client.post("/api/triage/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "esi_level" in data
    assert "acuity_label" in data
    assert data["esi_level"] in [1, 2, 3, 4, 5]
    assert "triage_agent" in data
    assert "diagnostic_agent" in data
    assert "pharmacist_agent" in data


def test_post_triage_analyze_image():
    """Test POST /api/triage/analyze-image returns vision diagnostic results."""
    payload = {
        "name": "Vision Patient",
        "age": 40,
        "gender": "Female",
        "chief_complaint": "Erythematous skin lesion",
        "symptoms": ["Erythema"],
        "vitals": {
            "heart_rate": 80,
            "bp_systolic": 120,
            "bp_diastolic": 80,
            "spo2": 98.0,
            "temperature_c": 37.0,
            "respiratory_rate": 16,
            "gcs": 15,
            "on_supplemental_o2": False
        },
        "medical_history": [],
        "current_medications": [],
        "allergies": [],
        "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSU5GGhAAAABJREFUeJzs0SERgDAUA8E9B6SgBAUoSgISUJA0M2H3s0x3v7v+c7m7y+3uLrf7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7v7vwB4xgEN7GgqEwAAAABJRU5ErkJggg=="
    }
    response = client.post("/api/triage/analyze-image", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "esi_level" in data
    assert data["vision_agent"] is not None
    assert data["vision_agent"]["agent_name"] == "Vision Diagnostic Agent"


def test_post_override():
    """Test POST /api/agents/override records clinician override."""
    override_payload = {
        "patient_id": "PT-TEST-100",
        "overridden_esi": 1,
        "clinician_id": "DR-SMITH",
        "notes": "Emergency physician manual override"
    }
    response = client.post("/api/agents/override", json=override_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
