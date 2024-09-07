import os, io, base64, json, random, pdb
from tqdm import tqdm
from openai import AzureOpenAI
from dotenv import load_dotenv
from PIL import Image,ImageFile
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

load_dotenv(override=True)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


def load_image_as_base64(image_path):
    image = Image.open(image_path).convert("RGB")
    if image.width > 512 or image.height > 512:
        # print("Resizing image")
        image = image.resize((512, 512))
        image_path = "/tmp/resized_image.png"
        image.save(image_path)
        
    with open(image_path, "rb") as image_file:
        # Read the image file in binary mode
        image_data = image_file.read()
        # Encode the image data to base64
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
        return base64_encoded
    
def get_text(image_path):
    json_file = "/".join(image_path.split('/')[:-1]) + '/content.json'
    with open(json_file) as fd:
        data = json.load(fd)
        text = data['Title'] +" "+ data['Description']
    return " ".join(text.split(" ")[:30])

def compare_images(image_path1, image_path2):
    base64_image1 = load_image_as_base64(image_path1)
    text1 = get_text(image_path1)

    base64_image2 = load_image_as_base64(image_path2)
    text2 = get_text(image_path2)

    message = f"To help you, I'm giving you also a short description of the images. Image 1: '{text1}' and Image 2: '{text2}'. Remember, you must return JUST the similarity score between 0 and 1"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an assistent that compares images encoded in base64 format. Return JUST the similarity score between 0 and 1 where 1 means the images are identical and 0 means they are completely different."
            },
            {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image1}",
                    }
                },
            ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image2}",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": message
            }
        ],
        max_tokens=200,
        top_p=0.1
    )

    return response.choices[0].message.content

def calculate_ssim(image1, image2):
    """Calculate SSIM between two PIL images."""
    image1 = Image.open(image1).convert('L')
    image2 = Image.open(image2).convert('L')
    if not (image1.width > 30 and image1.height > 30 and image2.width > 30 and image2.height > 30):
        assert False
    img1_np = np.array(image1.resize((512, 512)))
    img2_np = np.array(image2.resize((512, 512)))

    # Compute SSIM
    ssim_value, _ = ssim(img1_np, img2_np, full=True)
    assert float(ssim_value) < 0.1 or float(ssim_value) > 0.9
    return ssim_value

if __name__=="__main__":
    options = os.listdir("/Users/DToma/work/Aurora.Balthasar/images")
    dest = "/Users/DToma/work/image_similarity/pairs"
    os.makedirs(dest, exist_ok=True)
    needed = 60_000
    cnt = 0
    progress = tqdm(total=needed)
    pairs = []
    while True:
        if cnt == needed:
            break
        option1 = random.choice(options)
        option2 = random.choice(options)

        try:
            image = [i for i in os.listdir(f"/Users/DToma/work/Aurora.Balthasar/images/{option1}") if not i.endswith('.json')][0]
            image_path1 = f"/Users/DToma/work/Aurora.Balthasar/images/{option1}/{image}"
            image = [i for i in os.listdir(f"/Users/DToma/work/Aurora.Balthasar/images/{option2}") if not i.endswith('.json')][0]
            image_path2 = f"/Users/DToma/work/Aurora.Balthasar/images/{option2}/{image}"
            # similarity_score = compare_images(image_path1, image_path2)
            similarity_score = calculate_ssim(image_path1, image_path2)
            # print(f"Similarity score: {similarity_score}")
            similarity_score = float(similarity_score)
            assert 0 <= similarity_score <= 1
            rec = {"similarity_score": similarity_score, "image1": image_path1, "image2": image_path2}
            pairs.append(rec)
            cnt += 1
            progress.update(1)
        except Exception as e:
            continue
    progress.close()
    with open(f"{dest}/pairs.json", "w") as fd:
        json.dump(pairs, fd, indent=4)







