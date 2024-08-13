from pydantic import AnyUrl, BaseModel, Field


class ImageModel(BaseModel):
    url: AnyUrl = Field(
        title="URL",
        description="The web url of the file."
    )
