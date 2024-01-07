import easyocr
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium


def convert_pdf_to_images(file_path):
    pdf_file = pdfium.PdfDocument(file_path)
    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        # scale=scale,
    )

    list_final_images = []

    for i, image in zip(page_indices, renderer):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        list_final_images.append(dict({i: image_byte_array}))

    return list_final_images

def extract_text_with_easyocr(list_dict_final_images):
    language_reader = easyocr.Reader(['en'])
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):
        image = Image.open(BytesIO(image_bytes))
        raw_text = language_reader.readtext(image)
        raw_text = "\n".join([res[1] for res in raw_text])

        image_content.append(raw_text)

    return "\n".join(image_content)


if __name__ == "__main__":
    pdf_path = r"C:\Users\Darshan Joshi\Downloads\test_docs\test_docs\Precise Imaging Medical Records.pdf"  # Replace with the path to your PDF file
    extracted_text = extract_text_with_easyocr(convert_pdf_to_images(pdf_path))

    # Print or save the extracted text
    print(extracted_text)