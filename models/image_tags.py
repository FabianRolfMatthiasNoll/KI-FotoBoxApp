from pydantic import BaseModel


class ImageTags(BaseModel):
    Background: str
    Hats: str
    Glasses: str
    Effects: str
    Masks: str


class ImageTagsResponse(BaseModel):
    suggestion1: ImageTags
    suggestion2: ImageTags
    suggestion3: ImageTags
