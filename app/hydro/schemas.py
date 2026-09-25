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


def _validate_covariance3(value: object) -> object:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3 or any(
        not isinstance(row, list) or len(row) != 3 for row in value
    ):
        raise ValueError("covariance 必须是 3x3 矩阵，行/列顺序为 d18o、d2h、solute")
    for row in value:
        for cell in row:
            if not isinstance(cell, (int, float)) or cell != cell:
                raise ValueError("covariance 只能包含有限数值")
    for i in range(3):
        if value[i][i] < 0:
            raise ValueError("covariance 对角元（方差）不能为负")
        for j in range(i + 1, 3):
            if value[i][j] != value[j][i]:
                raise ValueError("covariance 必须对称")
    return value


class EndmemberCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    isotope_d18o: float = Field(..., ge=-100, le=100)
    isotope_d2h: float = Field(..., ge=-800, le=800)
    solute_mg_l: float = Field(..., ge=0, le=100000)
    # 未逐项指定标准差时，三个指标统一使用的 1σ 兜底值（与各指标同量纲）
    uncertainty: float = Field(default=0.1, gt=0, le=100)
    # 逐指标 1σ；为空的指标回退到 uncertainty
    isotope_d18o_std: float | None = Field(default=None, gt=0, le=1000)
    isotope_d2h_std: float | None = Field(default=None, gt=0, le=4000)
    solute_std: float | None = Field(default=None, gt=0, le=100000)
    # 完整 3x3 协方差（允许非零协方差项）；提供后覆盖逐指标标准差
    covariance: list[list[float]] | None = None
    version: str = Field(default="v1", min_length=1, max_length=40)

    @field_validator("covariance")
    @classmethod
    def _check_covariance(cls, value: object) -> object:
        return _validate_covariance3(value)


class SampleCreate(BaseModel):
    sample_code: str = Field(..., min_length=3, max_length=64)
    sampled_at: str = Field(..., min_length=20, max_length=40)
    isotope_d18o: float | None = Field(default=None, ge=-100, le=100)
    isotope_d2h: float | None = Field(default=None, ge=-800, le=800)
    solute_mg_l: float | None = Field(default=None, ge=0, le=100000)
    detection_limit: float = Field(default=0, ge=0, le=100000)
    # 未逐项指定标准差时的统一 1σ 兜底值（与各指标同量纲）
    measurement_error: float = Field(default=0.05, ge=0, le=100)
    isotope_d18o_std: float | None = Field(default=None, gt=0, le=1000)
    isotope_d2h_std: float | None = Field(default=None, gt=0, le=4000)
    solute_std: float | None = Field(default=None, gt=0, le=100000)
    covariance: list[list[float]] | None = None

    @field_validator("covariance")
    @classmethod
    def _check_covariance(cls, value: object) -> object:
        return _validate_covariance3(value)


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
