from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class LoanPayload(BaseModel):
    loan_id: str
    loan_amount: float = Field(gt=0)
    term_months: int = Field(gt=0, le=600)
    grade: str
    sub_grade: str
    int_rate: float = Field(ge=0, le=100)
    emp_length_years: Optional[float] = Field(default=None, ge=0, le=60)
    home_ownership: Literal["ANY", "MORTGAGE", "OWN", "RENT",
                             "any", "mortgage", "own", "rent"]
    annual_income: Optional[float] = Field(default=None, ge=0)
    verification_status: str
    purpose: str
    dti: Optional[float] = Field(default=None, description="-999 or null both mean missing")
    fico_range_low: int = Field(ge=300, le=850)
    fico_range_high: int = Field(ge=300, le=850)
    delinq_2yrs: int = Field(ge=0)
    pub_rec: int = Field(ge=0)
    open_acc: int = Field(ge=0)
    total_acc: int = Field(ge=0)
    revol_bal: float = Field(ge=0)
    revol_util: Optional[float] = Field(default=None, ge=0, le=100)
    collections_12_mths_ex_med: int = Field(ge=0)
    collateral_flag: Literal[0, 1]
    collateral_value: float = Field(ge=0)
    total_pymnt_before_default: float = Field(ge=0)
    outstanding_principal: float = Field(ge=0)
    days_past_due_at_default: int = Field(ge=0)

    @field_validator("fico_range_high")
    @classmethod
    def fico_high_ge_low(cls, v, info):
        lo = info.data.get("fico_range_low")
        if lo is not None and v < lo:
            raise ValueError("fico_range_high must be >= fico_range_low")
        return v

    @field_validator("dti")
    @classmethod
    def normalize_dti_sentinel(cls, v):
        # allow the legacy -999 sentinel through; clean() converts it to NaN downstream
        return v


class PredictionResponse(BaseModel):
    loan_id: str
    predicted_recovery_rate: float
    model_type: str
    model_version: str
