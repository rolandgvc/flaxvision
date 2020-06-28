from PIL import Image
import numpy as np
from flaxvision.models import vgg

rng = random.PRNGKey(0)

# load image
image = Image.open('fox.jpg')
image = image.resize((224, 224), Image.ANTIALIAS)
# image.show()

# convert to numpy
image = np.asarray(image)
image = image[np.newaxis, ...]

# load model
model = vgg.vgg11(rng=rng, pretrained=True)
pred = model(image)

print('class', np.argmax(pred))
print('score', np.max(pred))
