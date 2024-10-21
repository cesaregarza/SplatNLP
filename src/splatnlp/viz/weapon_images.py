import logging
from io import BytesIO

import requests
from PIL import Image, UnidentifiedImageError

from splatnlp.preprocessing.transform.mappings import generate_maps

logger = logging.getLogger(__name__)


def download_image(url: str) -> Image.Image:
    """Download an image from a given URL.

    Args:
        url (str): The URL of the image to download.

    Returns:
        Image.Image: The downloaded image as a PIL Image object, or None if the
            download fails.
    """
    logger.debug("Attempting to download image from %s", url)
    response = requests.get(url)
    try:
        image = Image.open(BytesIO(response.content))
        logger.debug("Successfully downloaded image from %s", url)
        return image
    except UnidentifiedImageError:
        logger.warning("Could not download image from %s", url)
        return None


def download_images() -> dict[str, tuple[str, Image.Image]]:
    """Download images for all weapons.

    Returns:
        dict[str, tuple[str, Image.Image]]: A dictionary where the key is the
            weapon ID, and the value is a tuple containing the weapon name and
            its corresponding image.
    """
    logger.info("Starting to download weapon images")
    _, id_to_name, id_to_url = generate_maps()
    images = {}
    total_weapons = len(id_to_url)
    for index, (weapon_id, url) in enumerate(id_to_url.items(), 1):
        logger.info(
            "Downloading image %d/%d for weapon ID: %s",
            index,
            total_weapons,
            weapon_id,
        )
        image = download_image(url)
        if image:
            images[weapon_id] = (id_to_name[weapon_id], image)
            logger.debug(
                "Successfully added image for weapon ID: %s", weapon_id
            )
        else:
            logger.warning("Failed to add image for weapon ID: %s", weapon_id)

    logger.info(
        "Finished downloading images. Total successful downloads: %d/%d",
        len(images),
        total_weapons,
    )
    return images
