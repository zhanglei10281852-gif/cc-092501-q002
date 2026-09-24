from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class WellCreate(BaseModel):
    code: str = Field(..., min_length=2, max_length=50)
    name: str = Field(..., min_length=1, max_length=120)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    aquifer: str = Field(..., min_length=1, max_length=120)
    screen_depth_m: float = Field(..., gt=0, le=5000)

    @field_validator("code")
    @classmethod
    def normalize_code(cls, value: str) -> str:
        return value.strip().upper()


class EndmemberCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    isotope_d18o: float = Field(..., ge=-100, le=100)
    isotope_d2h: float = Field(..., ge=-800, le=800)
    solute_mg_l: float = Field(..., ge=0, le=100000)
    uncertainty: float = Field(default=0.1, gt=0, le=100)
    version: str = Field(default="v1", min_length=1, max_length=40)


class SampleCreate(BaseModel):
    sample_code: str = Field(..., min_length=3, max_length=64)
    sampled_at: str = Field(..., min_length=20, max_length=40)
    isotope_d18o: float | None = Field(default=None, ge=-100, le=100)
    isotope_d2h: float | None = Field(default=None, ge=-800, le=800)
    solute_mg_l: float | None = Field(default=None, ge=0, le=100000)
    detection_limit: float = Field(default=0, ge=0, le=100000)
    measurement_error: float = Field(default=0.05, ge=0, le=100)


class InversionRequest(BaseModel):
    endmember_ids: list[int] = Field(..., min_length=2, max_length=8)
    method: str = Field(default="weighted-least-squares", pattern="^(weighted-least-squares|projected-gradient)$")
    max_iterations: int = Field(default=500, ge=10, le=10000)
    tolerance: float = Field(default=1e-8, gt=0, le=0.1)
    model_version: str = Field(default="mix-1", min_length=1, max_length=40)


class TransportRequest(BaseModel):
    source_concentration: float = Field(..., ge=0, le=1000000)
    distance_m: float = Field(..., gt=0, le=1000000)
    velocity_m_day: float = Field(..., gt=0, le=10000)
    dispersion_m2_day: float = Field(..., gt=0, le=100000)
    decay_per_day: float = Field(default=0, ge=0, le=100)
    duration_days: float = Field(..., gt=0, le=100000)
    step_days: float = Field(default=1, gt=0, le=1000)
    model_version: str = Field(default="ade-1", min_length=1, max_length=40)

