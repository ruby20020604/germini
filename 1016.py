import vertexai
from vertexai.preview.vision_models import Image, ImageGenerationModel
from google.oauth2 import service_account
from PIL import Image
import IPython.display as display


credentials = service_account.Credentials.from_service_account_file("/Users/wangyunruo/credenital.json")

vertexai.init(project=credentials.project_id, location="us-central1", credentials=credentials)

prompt = "a men and a women in Taiwan"

model = ImageGenerationModel.from_pretrained("imagegeneration@006")

response = model.generate_images(
    prompt=prompt,
    number_of_images = 1,
    language="en",
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    person_generation="allow_adult",
    #negative_prompt="no inside the house"
)

len(response.images)

response.images[0].show()

#image = response.images[0]
#image.save("generated_image.png")
#display.Image(filename="generated_image.png")
