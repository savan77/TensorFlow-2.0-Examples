import tensorflow as tf
import os
import matplotlib.pyplot as plt
import glob

def load_image(image_path, width=512, height=512):
	img_raw = tf.io.read_file(image_path)
	img = tf.image.deoce_jpeg(img_raw, channels=3)
	img = tf.image.resize(img, [width, height])
	return img

def read_data(IN_DIR, dir_list):
	for dir_ in dir_list:
		img_paths = glob.glob(os.path.join(IN_DIR, dir_) + "/*.jpg")
		img_paths.extend(glob.glob(os.path.join(IN_DIR, dir_) + "/*.jpeg"))
		img_paths.extend(glob.glob(os.path.join(IN_DIR, dir_) + "/*.png"))
	paths = tf.data.Dataset.from_tensor_slices(img_paths)
	img_loader = paths.map(load_image, num_parallel_calls=AUTOTUNE)




def visualize_image(image):
	plt.imshow(image)
	plt.grid(False)
	plt.show()


def main():
	IN_DIR ="./dataset/"
	dir_list = next(os.walk(IN_DIR))[1]  #also labels list
	print("Found {} lables...".format(len(dir_list)))
	print("Reading dataset...")
	read_data(IN_DIR, dir_list)

if __name__ == "__main__":
	main()
