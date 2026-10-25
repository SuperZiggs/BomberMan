from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

VERSION = 1


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["play", "watch"]
    model_id: str = Field(min_length=1, max_length=128)
    num_agents: StrictInt = Field(ge=2, le=4)
    board_size: StrictInt = 11
    speed: StrictInt = Field(default=12, ge=1, le=60)
    seed: StrictInt | None = Field(default=None, ge=0, le=2**32 - 1)
    max_steps: StrictInt = Field(default=500, ge=1, le=5000)
    sudden_death: bool = False

    @model_validator(mode="after")
    def supported(self):
        if self.board_size not in (9, 11):
            raise ValueError("Bản đồ hỗ trợ kích thước 9 × 9 hoặc 11 × 11.")
        if self.mode == "play" and self.num_agents not in (3, 4):
            raise ValueError("Chế độ Chơi hỗ trợ 3 hoặc 4 tác tử.")
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phase: Literal["C", "D", "E"]
    episodes: StrictInt = Field(ge=1, le=100000)
    num_agents: StrictInt = Field(ge=2, le=4)
    warm_start: bool = False
    model_id: str | None = None
    max_steps: StrictInt = Field(default=500, ge=1, le=500)

    @model_validator(mode="after")
    def supported(self):
        if self.phase in ("C", "D") and self.num_agents != 2:
            raise ValueError("Phase C/D yêu cầu đúng 2 tác tử.")
        if self.warm_start and not self.model_id:
            raise ValueError("Hãy chọn checkpoint để warm-start.")
        return self
