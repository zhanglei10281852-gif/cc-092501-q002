from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator, model_validator


def _validate_covariance3(value: list[list[float]]) -> list[list[float]]:
    """校验 3×3 协方差：形状、对称、正方差、相关系数不越界。"""
    if len(value) != 3 or any(len(row) != 3 for row in value):
        raise ValueError("covariance 必须为 3×3 矩阵")
    for i in range(3):
        if value[i][i] <= 0:
            raise ValueError("协方差对角元素（方差）必须为正")
        for j in range(i + 1, 3):
            if not math.isclose(value[i][j], value[j][i], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("协方差矩阵必须对称")
            bound = math.sqrt(value[i][i] * value[j][j])
            if abs(value[i][j]) > bound * (1 + 1e-9):
                raise ValueError(f"协方差元素 [{i}][{j}] 超出相关系数 |ρ|≤1 的范围")
    return value


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
    # 逐指标测量标准差（缺省回退到统一 uncertainty）
    sigma_d18o: float | None = Field(default=None, gt=0, le=10000)
    sigma_d2h: float | None = Field(default=None, gt=0, le=10000)
    sigma_solute: float | None = Field(default=None, gt=0, le=10000)
    # 端元三指标完整协方差矩阵（行/列顺序：δ18O、δ2H、溶质）
    covariance: list[list[float]] | None = Field(default=None)
    version: str = Field(default="v1", min_length=1, max_length=40)

    @model_validator(mode="after")
    def _check_covariance(self) -> "EndmemberCreate":
        if self.covariance is not None:
            _validate_covariance3(self.covariance)
        return self


class SampleCreate(BaseModel):
    sample_code: str = Field(..., min_length=3, max_length=64)
    sampled_at: str = Field(..., min_length=20, max_length=40)
    isotope_d18o: float | None = Field(default=None, ge=-100, le=100)
    isotope_d2h: float | None = Field(default=None, ge=-800, le=800)
    solute_mg_l: float | None = Field(default=None, ge=0, le=100000)
    detection_limit: float = Field(default=0, ge=0, le=100000)
    measurement_error: float = Field(default=0.05, ge=0, le=100)
    # 逐指标观测标准差（缺省回退到统一 measurement_error）
    sigma_d18o: float | None = Field(default=None, gt=0, le=10000)
    sigma_d2h: float | None = Field(default=None, gt=0, le=10000)
    sigma_solute: float | None = Field(default=None, gt=0, le=10000)
    # 观测三指标完整协方差矩阵（行/列顺序：δ18O、δ2H、溶质）
    covariance: list[list[float]] | None = Field(default=None)

    @model_validator(mode="after")
    def _check_covariance(self) -> "SampleCreate":
        if self.covariance is not None:
            _validate_covariance3(self.covariance)
        return self


class ObservationWeights(BaseModel):
    """反演请求级权重覆盖，仅作用于本次任务，不回写样本/端元档案。"""

    sigma_d18o: float | None = Field(default=None, gt=0, le=10000)
    sigma_d2h: float | None = Field(default=None, gt=0, le=10000)
    sigma_solute: float | None = Field(default=None, gt=0, le=10000)
    covariance: list[list[float]] | None = Field(default=None)

    @model_validator(mode="after")
    def _check_covariance(self) -> "ObservationWeights":
        if self.covariance is not None:
            _validate_covariance3(self.covariance)
        return self


class EndmemberWeightOverride(BaseModel):
    endmember_id: int
    sigma_d18o: float | None = Field(default=None, gt=0, le=10000)
    sigma_d2h: float | None = Field(default=None, gt=0, le=10000)
    sigma_solute: float | None = Field(default=None, gt=0, le=10000)
    covariance: list[list[float]] | None = Field(default=None)

    @model_validator(mode="after")
    def _check_covariance(self) -> "EndmemberWeightOverride":
        if self.covariance is not None:
            _validate_covariance3(self.covariance)
        return self


class InversionRequest(BaseModel):
    endmember_ids: list[int] = Field(..., min_length=2, max_length=8)
    method: str = Field(
        default="weighted-least-squares",
        pattern="^(weighted-least-squares|projected-gradient)$",
    )
    max_iterations: int = Field(default=500, ge=10, le=10000)
    tolerance: float = Field(default=1e-8, gt=0, le=0.1)
    model_version: str = Field(default="mix-1", min_length=1, max_length=40)
    observation_weights: ObservationWeights | None = None
    endmember_weights: list[EndmemberWeightOverride] | None = None

    @model_validator(mode="after")
    def _check_endmember_weight_ids(self) -> "InversionRequest":
        if self.endmember_weights is not None:
            override_ids = [item.endmember_id for item in self.endmember_weights]
            if len(set(override_ids)) != len(override_ids):
                raise ValueError("端元权重覆盖中 endmember_id 不能重复")
            unknown = set(override_ids) - set(self.endmember_ids)
            if unknown:
                raise ValueError(f"端元权重覆盖引用了未参与反演的端元: {sorted(unknown)}")
        return self


class TransportRequest(BaseModel):
    source_concentration: float = Field(..., ge=0, le=1000000)
    distance_m: float = Field(..., gt=0, le=1000000)
    velocity_m_day: float = Field(..., gt=0, le=10000)
    dispersion_m2_day: float = Field(..., gt=0, le=100000)
    decay_per_day: float = Field(default=0, ge=0, le=100)
    duration_days: float = Field(..., gt=0, le=100000)
    step_days: float = Field(default=1, gt=0, le=1000)
    model_version: str = Field(default="ade-1", min_length=1, max_length=40)
