from typing import Literal
from pydantic import BaseModel


class ImageTags(BaseModel):
    Background: Literal["beach", "sunset", "space", "none"]
    Hats: Literal[
        "party_hat",
        "crown",
        "cowboy_hat",
        "astronaut",
        "none",
    ]
    Glasses: Literal[
        "glasses_clown_nose",
        "glasses_googley_eyes",
        "glasses_nose_mustache",
        "glasses_heart",
        "none",
    ]
    Effects: Literal["confetti", "sparkles", "spotlight", "hearts", "none"]
    Masks: Literal["astronaut_helmet", "none"]


class ImageTagsResponse(BaseModel):
    suggestion1: ImageTags
    suggestion2: ImageTags
    suggestion3: ImageTags
