import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 150, 150
model_path = 'C:/Users/dell/Desktop/AIrizal/models/model.h5'
model_weights_path = 'C:/Users/dell/Desktop/AIrizal/models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Label: Calinawan Cave")
  elif answer == 1:
    print("Label: Daraitan")
  elif answer == 2:
    print("Label: Daranak")
  elif answer == 3:
    print("Label: Hapunang Banoi")
  elif answer == 4:
    print("Label: Kinabuan Falls")
  elif answer == 5:
    print("Label: Mt. Lagyo")
  elif answer == 6:
    print("Label: Mt. Oro")
  elif answer == 7:
    print("Label: Mystical Cave")
  elif answer == 8:
    print("Label: Palo Alto")
  elif answer == 9:
    print("Label: Tungtong Falls")
  elif answer == 10:
    print("Label: Wawa dam")

  return answer

Calinawan_Cave_t = 0
Calinawan_Cave_f = 0
Daraitan_t = 0
Daraitan_f = 0
Daranak_t = 0
Daranak_f = 0
Hapunang_Banoi_t = 0
Hapunang_Banoi_f = 0
Kinabuan_Falls_t = 0
Kinabuan_Falls_f = 0
Mt_Balagbag_t = 0
Mt_Balagbag_f = 0
Mt_Lagyo_t = 0
Mt_Lagyo_f = 0
Mt_Oro_t = 0
Mt_Oro_f = 0
Mystical_Cave_t = 0
Mystical_Cave_f = 0
Palo_Alto_t = 0
Palo_Alto_f = 0
Tungtong_Falls_t = 0
Tungtong_Falls_f = 0
Wawa_dam_t = 0
Wawa_dam_f = 0



for i, ret in enumerate(os.walk('C:/Users/dell/desktop/AIrizal/test-data/Calinawan Cave')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue

    result = predict(ret[0] + '/' + filename)
    if result == 0:
      Calinawan_Cave_t += 1
    else:
      Calinawan_Cave_f += 1

for i, ret in enumerate(os.walk('C:/Users/dell/desktop/AIrizal/test-data/Daraitan')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue

    result = predict(ret[0] + '/' + filename)
    if result == 1:
      Daraitan_t += 1
    else:
      Daraitan_f += 1
      
for i, ret in enumerate(os.walk('C:/Users/dell/desktop/AIrizal/test-data/Daranak')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue

    result = predict(ret[0] + '/' + filename)
    if result == 2:
      Daranak_t += 1
    else:
      Daranak_f += 1

for i, ret in enumerate(os.walk('C:/Users/dell/desktop/AIrizal/test-data/Hapunang Banoi')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue

    result = predict(ret[0] + '/' + filename)
    if result == 3:
      print(ret[0] + '/' + filename)
      Hapunang_Banoi_t += 1
    else:
      Hapunang_Banoi_f += 1
     
for i, ret in enumerate(os.walk('C:/Users/dell/desktop/AIrizal/test-data/Kinabuan Falls')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue

    result = predict(ret[0] + '/' + filename)
    if result == 4:
      print(ret[0] + '/' + filename)
      Kinabuan_Falls_t += 1
    else:
      Kinabuan_Falls_f += 1

      
for i, ret in enumerate(os.walk('C:/Users/dell/desktop/AIrizal/test-data/Mt. Lagyo')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue

    result = predict(ret[0] + '/' + filename)
    if result == 5:
      print(ret[0] + '/' + filename)
      Mt_Lagyo_t += 1
    else:
      Mt_Lagyo_f += 1

for i, ret in enumerate(os.walk('C:/Users/dell/desktop/AIrizal/test-data/Mt. Oro')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue

    result = predict(ret[0] + '/' + filename)
    if result == 6:
      print(ret[0] + '/' + filename)
      Mt_Oro_t += 1
    else:
      Mt_Oro_f += 1

for i, ret in enumerate(os.walk('C:/Users/dell/desktop/AIrizal/test-data/Mystical Cave')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue

    result = predict(ret[0] + '/' + filename)
    if result == 7:
      print(ret[0] + '/' + filename)
      Mystical_Cave_t += 1
    else:
      Mystical_Cave_f += 1

for i, ret in enumerate(os.walk('C:/Users/dell/desktop/AIrizal/test-data/Palo Alto')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    result = predict(ret[0] + '/' + filename)
    if result == 8:
      print(ret[0] + '/' + filename)
      Palo_Alto_t += 1
    else:
      Palo_Alto_f += 1

for i, ret in enumerate(os.walk('C:/Users/dell/desktop/AIrizal/test-data/Tungtong Falls')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Sunflower")
    result = predict(ret[0] + '/' + filename)
    if result == 9:
      print(ret[0] + '/' + filename)
      Tungtong_Falls_t += 1
    else:
      Tungtong_Falls_f += 1

for i, ret in enumerate(os.walk('C:/Users/dell/desktop/AIrizal/test-data/Wawa Dam')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue

    result = predict(ret[0] + '/' + filename)
    if result == 10:
      print(ret[0] + '/' + filename)
      Wawa_dam_t += 1
    else:
      Wawa_dam_f += 1


"""
Check metrics
"""
print("True Calinawan Cave: ", Calinawan_Cave_t)
print("False Calinawan Cave: ", Calinawan_Cave_f)
print("True Daraitan: ", Daraitan_t)
print("False Daraitan: ", Daraitan_f)
print("True Daranak: ", Daranak_t)
print("False Daranak: ", Daranak_f)
print("True Hapunang Banoi: ",  Hapunang_Banoi_t)
print("False Hapunang Banoi: ",  Hapunang_Banoi_f)
print("True Kinabuan Falls: ",  Kinabuan_Falls_t)
print("False Kinabuan Falls: ", Kinabuan_Falls_f)
print("True Mt. Lagyo: ", Mt_Lagyo_t)
print("False Mt. Lagyo: ", Mt_Lagyo_f)
print("True Mt. Oro: ",  Mt_Oro_t)
print("False Mt. Oro: ",  Mt_Oro_f)
print("True Mystical Cave: ", Mystical_Cave_t)
print("False Mystical Cave: ", Mystical_Cave_f)
print("True Palo Alto: ", Palo_Alto_t)
print("False Palo Alto: ", Palo_Alto_f)
print("True Wawa dam: ",  Wawa_dam_t)
print("False Wawa dam: ",  Wawa_dam_f)
