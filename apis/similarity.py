from fastapi import APIRouter

from io_controller import IOController
from manager import ImageSearchManager
from models import ImageModel
from settings import settings

router = APIRouter()

@router.post("/similar", response_model=list[ImageModel])
async def get_similar_images(image: ImageModel) -> list[ImageModel]:
    production_image_dir_path = settings.production_image_dir_path
    image_file_path = IOController.download_image_file(
        url=image.url, base_path=production_image_dir_path)
    similar_image_urls = ImageSearchManager.get_n_most_similar_urls(
        image_file_path=image_file_path)
    similar_images = [None] * len(similar_image_urls)
    for index, similar_image_url in enumerate(similar_image_urls):
        similar_images[index] = ImageModel(url=similar_image_url)
    return similar_images
