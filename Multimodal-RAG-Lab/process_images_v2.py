import os
import re
import fitz
import base64
import asyncio
import mimetypes
from PIL import Image
from mimetypes import guess_type
from aiohttp import ClientSession
from openai import AsyncAzureOpenAI
from azure.storage.blob.aio import BlobServiceClient


# Calculate the bounding box from single polygon
def calculate_bounding_box(polygon):
    # Separate the polygons
    absolute_polygon = [(polygon[i] * 72, polygon[i + 1] * 72)
                        for i in range(0, len(polygon), 2)]

    # Calculate the bounding box
    min_x = min(x for x, _ in absolute_polygon)
    max_x = max(x for x, _ in absolute_polygon)

    # The top of the bounding box is the bottom of the first polygon
    min_y = min(y for _, y in absolute_polygon)
    max_y = max(y for _, y in absolute_polygon)

    bounding_box = [min_x, min_y, max_x, max_y]
    return bounding_box


# Calculate the bounding box from two polygons
def calculate_bounding_box_m(polygons):
    # Separate the polygons
    polygon1, polygon2 = polygons
    absolute_polygon1 = [(polygon1[i] * 72, polygon1[i + 1] * 72)
                         for i in range(0, len(polygon1), 2)]
    absolute_polygon2 = [(polygon2[i] * 72, polygon2[i + 1] * 72)
                         for i in range(0, len(polygon2), 2)]

    # Calculate the bounding box
    min_x = min(min(x for x, _ in absolute_polygon1), min(
        x for x, _ in absolute_polygon2))
    max_x = max(max(x for x, _ in absolute_polygon1), max(
        x for x, _ in absolute_polygon2))
    min_y = min(min(y for _, y in absolute_polygon1), min(
        y for _, y in absolute_polygon2))
    max_y = max(max(y for _, y in absolute_polygon1), max(
        y for _, y in absolute_polygon2))

    bounding_box = [min_x, min_y, max_x, max_y]
    return bounding_box


def crop_image_from_pdf_page2(pdf_path, page_number, bounding_box):
    """
    Crops a region from a given page in a PDF and returns it as an image.

    :param pdf_path: Path to the PDF file.
    :param page_number: The page number to crop from (0-indexed).
    :param bounding_box: A tuple of (x0, y0, x1, y1) coordinates for the bounding box.
    :return: A PIL Image of the cropped area.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)

    # Increase resolution (e.g., 300 DPI)
    zoom_x = 214 / 72  # Horizontal zoom factor
    zoom_y = 214 / 72  # Vertical zoom factor
    mat = fitz.Matrix(zoom_x, zoom_y)

    # Cropping the page. The rect requires the coordinates in the format (x0, y0, x1, y1).
    rect = fitz.Rect(bounding_box)
    cropped_page = page.get_pixmap(matrix=mat, clip=rect)

    # Saving the cropped area to an image
    cropped_image_path = 'data/cropped_image_page2.png'
    cropped_page.save(cropped_image_path)

    doc.close()
    return cropped_image_path


def crop_image_from_image(image_path, page_number, bounding_box):
    """
    Crops an image based on a bounding box.

    :param image_path: Path to the image file.
    :param page_number: The page number of the image to crop (for TIFF format).
    :param bounding_box: A tuple of (left, upper, right, lower) coordinates for the bounding box.
    :return: A cropped image.
    :rtype: PIL.Image.Image
    """
    with Image.open(image_path) as img:
        if img.format == "TIFF":
            # Open the TIFF image
            img.seek(page_number)
            img = img.copy()

        # The bounding box is expected to be in the format (left, upper, right, lower).
        cropped_image = img.crop(bounding_box)
        return cropped_image


def crop_image_from_pdf_page(pdf_path, page_number, bounding_box):
    """
    Crops a region from a given page in a PDF and returns it as an image.

    :param pdf_path: Path to the PDF file.
    :param page_number: The page number to crop from (0-indexed).
    :param bounding_box: A tuple of (x0, y0, x1, y1) coordinates for the bounding box.
    :return: A PIL Image of the cropped area.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)

    # Cropping the page. The rect requires the coordinates in the format (x0, y0, x1, y1).
    bbx = [x * 72 for x in bounding_box]
    rect = fitz.Rect(bbx)
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), clip=rect)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    doc.close()

    return img


def crop_image_from_file(file_path, page_number, bounding_box):
    """
    Crop an image from a file.

    Args:
        file_path (str): The path to the file.
        page_number (int): The page number (for PDF and TIFF files, 0-indexed).
        bounding_box (tuple): The bounding box coordinates in the format (x0, y0, x1, y1).

    Returns:
        A PIL Image of the cropped area.
    """
    mime_type = mimetypes.guess_type(file_path)[0]

    if mime_type == "application/pdf":
        return crop_image_from_pdf_page(file_path, page_number, bounding_box)
    else:
        return crop_image_from_image(file_path, page_number, bounding_box)


async def upload_file(file_path: str, containerName: str, file_prefix: str):
    try:
        # Initialize the BlobServiceClient with the connection string
        async with BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING")) as storage_client:
            blob_container_client = storage_client.get_container_client(
                containerName)

            # Read the local file
            with open(file_path, "rb") as file:
                file_name = os.path.basename(file_path)

                # Upload the file to the container
                file_blob_name = f"{file_prefix}/{file_name}"
                await blob_container_client.upload_blob(name=file_blob_name, data=file, overwrite=True)

            # print(f"File {file_name} uploaded successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(
            image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


async def understand_image_with_gptv(image_path, caption=None):
    """
    Generates a description for an image using the GPT-4V model.

    Parameters:
    - api_base (str): The base URL of the API.
    - api_key (str): The API key for authentication.
    - deployment_name (str): The name of the deployment.
    - api_version (str): The version of the API.
    - image_path (str): The path to the image file.
    - caption (str): The caption for the image.

    Returns:
    - img_description (str): The generated description for the image.
    """

    system_prompt = """
    # Role
    In this scenario, your role is an advanced image reader to capture information from images. Make sure you fully understand the given image and extract all detailed information..

    # Tasks
    Your core responsibility is to extract detailed information from the image, including all texts and the semantics.

    # Constraints
    - Quality Assurance: Make sure you fully understand the given image and extract all detailed information.
    - DON'T make up any information.
    """

    deployment_name = os.getenv('AZURE_OPENAI_MODEl_GPT_4o_mini')

    client = AsyncAzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY_GPT_4o_mini'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION_GPT_4o_mini'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT_GPT_4o_mini')
    )

    data_url = local_image_to_data_url(image_path)
    async with ClientSession():
        response = await client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": f"Describe this image (note: it has image caption: {caption}):" if caption else "Describe this image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        img_description = response.choices[0].message.content
        return img_description


def update_figure_description(md_content: str, img_description: str, idx: int, containerName: str, folder: str):
    """
    Updates the figure description in the Markdown content.

    Args:
        md_content (str): The original Markdown content.
        img_description (str): The new description for the image.
        idx (int): The index of the figure.

    Returns:
        str: The updated Markdown content with the new figure description.
    """

    # The substring you're looking for
    figure_positions = find_figure_positions(md_content)

    storage_account = ""
    match = re.search(r'AccountName=([^;]+)', os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    if match:
        storage_account = match.group(1)
    else:
        return

    img_url = f"blob://{storage_account}/{containerName}/{folder}/figure_{idx}.png"

    new_string = f"<figure>\n\n![]({img_url})<!-- FigureContent=\"{img_description}\" -->\n\n</figure>"

    new_md_content = md_content

    start_index = figure_positions[idx][0]
    end_index = figure_positions[idx][1]
    if start_index != -1 and end_index != -1:
        # Replace the old string with the new string
        new_md_content = md_content[:start_index] + \
            new_string + md_content[end_index:]

    return new_md_content


def get_bounding_box(polygon):
    """
    Calculate the bounding box for a given polygon.

    Args:
        polygon (list): A list of coordinates in the form [x0, y0, x1, y1, ..., xn, yn].

    Returns:
        tuple: A tuple containing (min_x, min_y, max_x, max_y).
    """
    # Initialize min and max values
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    # Iterate through the polygon points
    for i in range(0, len(polygon), 2):
        x = polygon[i]
        y = polygon[i + 1]
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    return (min_x, min_y, max_x, max_y)


def find_figure_positions(input_html: str):
    pattern = re.compile(r"<figure>.*?</figure>", re.DOTALL | re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(input_html)]


def find_table_positions(input_html: str):
    pattern = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(input_html)]


def find_pagebreak_positions(input_html: str):
    pattern = re.compile(r"<!-- PageBreak -->", re.DOTALL | re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(input_html)]


def find_chunk_pages(pagebreaks, start_index, end_index):
    # Initialize a list to store the overlapping page breaks
    overlapping_page_numbers = []
    
    # Calculate the adjusted page ranges
    adjusted_pagebreaks = []
    last_end = -1
    for _, end in pagebreaks:
        if last_end == -1:
            adjusted_pagebreaks.append((0, end))
        else:
            adjusted_pagebreaks.append((last_end + 1, end))
        last_end = end
    
    # Check for overlaps with the adjusted page ranges
    for i, (page_start, page_end) in enumerate(adjusted_pagebreaks):
        if (start_index <= page_end and end_index >= page_start):
            overlapping_page_numbers.append(i + 1)  # Page numbers are 1-indexed
    
    return overlapping_page_numbers


async def include_figures_in_md(input_file_path, result, containerName, folder, output_folder="data/cropped"):
    md_content = result.content
    image_tasks = []
    img_descriptions = []

    # fig_metadata = {}
    if result.figures:
        # print("Figures:")
        for idx, figure in enumerate(result.figures):
            figure_content = ""
            for _, span in enumerate(figure.spans):
                figure_content += md_content[span.offset:span.offset + span.length]

            if figure.caption:
                caption_region = figure.caption.bounding_regions
                for region in figure.bounding_regions:
                    if region not in caption_region:
                        bounding_box = get_bounding_box(region.polygon)
                        cropped_image = crop_image_from_file(
                            input_file_path, region.page_number - 1, bounding_box)

                        output_file = f"figure_{idx}.png"
                        cropped_image_filename = os.path.join(
                            output_folder, output_file)

                        cropped_image.save(cropped_image_filename)
                        await upload_file(cropped_image_filename, containerName, folder)
                        image_tasks.append(understand_image_with_gptv(
                            cropped_image_filename, figure.caption.content))
            else:
                for region in figure.bounding_regions:
                    bounding_box = get_bounding_box(region.polygon)

                    cropped_image = crop_image_from_file(
                        input_file_path, region.page_number - 1, bounding_box)

                    output_file = f"figure_{idx}.png"
                    cropped_image_filename = os.path.join(
                        output_folder, output_file)
                    cropped_image.save(cropped_image_filename)
                    await upload_file(cropped_image_filename, containerName, folder)
                    image_tasks.append(understand_image_with_gptv(
                        cropped_image_filename, ""))

            if len(image_tasks) >= 10:
                results = await asyncio.gather(*image_tasks)
                img_descriptions.extend(results)
                image_tasks = []

        if image_tasks:
            results = await asyncio.gather(*image_tasks)
            img_descriptions.extend(results)

        for idx, img_desc in enumerate(img_descriptions):
            md_content = update_figure_description(
                md_content, img_desc, idx, containerName, folder)

    return md_content
