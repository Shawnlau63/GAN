import os

path = r'E:\AI\GAN\faces'
# print(os.listdir(r'E:\AI\GAN\faces'))

dataset = []
li = os.listdir(path)
print(li)
dataset.extend(os.path.join(path,x) for x in li)
print(dataset)