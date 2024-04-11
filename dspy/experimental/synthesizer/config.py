from typing import Any, Optional

from pydantic import BaseModel, model_validator


class SynthesizerArguments(BaseModel):
    feedback_mode: Optional[str] = None
    num_example_for_feedback: Optional[int] = None

    input_lm_model: Optional[Any] = None
    output_lm_model: Optional[Any] = None
    output_teacher_module: Optional[Any] = None

    num_example_for_optim: Optional[int] = None

    @model_validator(mode="after")
    def validate_feedback_mode(self):
        if not self.feedback_mode:
            return self
        feedback_mode_valid = self.feedback_mode in ["human", "llm"]
        if not feedback_mode_valid:
            raise ValueError("Feedback mode should be either 'human' or 'llm'.")

        elif not self.num_example_for_feedback:
            raise ValueError(
                "Number of examples for feedback is required when feedback mode is provided."
            )

        return self
