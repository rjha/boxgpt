from typing import Any
from pydantic import BaseModel, Field, field_validator



class SheetKeyMap(BaseModel):
    """Deterministic row and column key mapping deduced from sheet structure."""

    sheet_title: str = Field(
        default="",
        description="Top-level sheet banner title joined with ' | '.",
    )
    col_keys: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of Excel column letter to its hierarchical header key using exact sheet labels joined by '::'.",
    )
    row_keys: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of Excel row number (as string) to its hierarchical row key using exact sheet labels joined by '::'.",
    )
    hidden_rows: list[int] = Field(
        default_factory=list,
        description="Row numbers marked [HIDDEN] in the skeleton.",
    )
    ignore_rows: list[int] = Field(
        default_factory=list,
        description="Banner rows, header rows, blank rows, and header-only rows without values.",
    )
    outlier_cells: dict[str, str] = Field(
        default_factory=dict,
        description="Optional mapping of off-grid coordinates to namespaced keys.",
    )

    @field_validator("outlier_cells", mode="before")
    @classmethod
    def _normalize_outlier_cells(cls, v: Any) -> Any:
        if v == []:
            return {}
        return v
