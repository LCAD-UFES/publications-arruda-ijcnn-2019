import zipfile
import shutil
import os
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import json 
import sys

images_folder = 'bdd100k_images'
labels_folder = 'bdd100k_labels'


img_train_dir = os.path.join(images_folder, 'bdd100k', 'images', '100k', 'train')
lab_train_dir = os.path.join(labels_folder, 'bdd100k', 'labels', '100k', 'train')

img_val_dir = os.path.join(images_folder, 'bdd100k', 'images', '100k', 'val')
lab_val_dir = os.path.join(labels_folder, 'bdd100k', 'labels', '100k', 'val')


def extract_here(zipped, target):

	print('Extracting file {} to {} ...'.format(zipped, target), end='', flush=True)
	
	with zipfile.ZipFile(zipped, 'r') as zip_ref:
		zip_ref.extractall(target)

	print(' Done !')


def move_from_to_list(list_path, img_to_dir, lab_to_dir):

	print('Getting images and labels from list {}:'.format(list_path))

	with open(list_path, 'r') as file:
		images = file.readlines()

	images = [i.strip() for i in images]

	for i in tqdm(images):

		try:
			shutil.move(os.path.join(img_train_dir, i), os.path.join(img_to_dir, i))
			shutil.move(os.path.join(lab_train_dir, i[:-3] + 'json'), os.path.join(lab_to_dir, i[:-3] + 'json'))
		except:
			shutil.move(os.path.join(img_val_dir, i), os.path.join(img_to_dir, i))
			shutil.move(os.path.join(lab_val_dir, i[:-3] + 'json'), os.path.join(lab_to_dir, i[:-3] + 'json'))



extract_here('bdd100k_images.zip', images_folder)
extract_here('bdd100k_labels.zip', labels_folder)


img_train_day_dir_to = os.path.join('images', 'train', 'day')
img_train_night_dir_to = os.path.join('images', 'train', 'night')

img_test_day_dir_to = os.path.join('images', 'test', 'day')
img_test_night_dir_to = os.path.join('images', 'test', 'night')

lab_train_day_dir_to = os.path.join('labels', 'train', 'day')
lab_train_night_dir_to = os.path.join('labels', 'train', 'night')

lab_test_day_dir_to = os.path.join('labels', 'test', 'day')
lab_test_night_dir_to = os.path.join('labels', 'test', 'night')


os.makedirs(img_train_day_dir_to)
os.makedirs(img_train_night_dir_to)
os.makedirs(img_test_day_dir_to)
os.makedirs(img_test_night_dir_to)

os.makedirs(lab_train_day_dir_to)
os.makedirs(lab_train_night_dir_to)
os.makedirs(lab_test_day_dir_to)
os.makedirs(lab_test_night_dir_to)


move_from_to_list(os.path.join('lists', 'train_day.txt'), img_train_day_dir_to, lab_train_day_dir_to)
move_from_to_list(os.path.join('lists', 'train_night.txt'), img_train_night_dir_to, lab_train_night_dir_to)

move_from_to_list(os.path.join('lists', 'test_day.txt'), img_test_day_dir_to, lab_test_day_dir_to)
move_from_to_list(os.path.join('lists', 'test_night.txt'), img_test_night_dir_to, lab_test_night_dir_to)


shutil.rmtree(images_folder)
shutil.rmtree(labels_folder)


side = 256
current_data = None
json_dir = None
without_car = 0

def _in_rule(width, height, cropped, occluded, truncated):

	if truncated or cropped or occluded:
		return width > 30.0 and height > 30.0
	else:
		return width > 20.0 and height > 20.0


##### JSON CROP AND RESIZE #####

def _adjust_car_box2d(filename, sw, h, savefile=True):

	global current_data

	with open(os.path.join(json_dir, filename + '.json'), 'r') as f:
		data = json.load(f)

	x = []

	for obj in data['frames'][0]['objects']:
		tmp_debug_x = []
		if obj['category'] == 'car':
			x_pts = [obj['box2d']['x1'], obj['box2d']['x2']]
			x1, x2 = min(x_pts), max(x_pts)	

			new_x1 = max(sw, min(sw + h, x1))
			new_x2 = max(sw, min(sw + h, x2))

			cropped = new_x1 != x1 or new_x2 != x2
			occluded = obj['attributes']['occluded']
			truncated = obj['attributes']['truncated']

			new_x1 -= sw
			new_x2 -= sw

			new_pts = [new_x1, obj['box2d']['y1'], new_x2, obj['box2d']['y2']]
			new_pts = [(side * e)/h for e in new_pts] # rescale to (side, side)
			new_pts = [max(0, min(e, side)) for e in new_pts]

			new_pts = [float(i) for i in new_pts]

			width = new_pts[2] - new_pts[0]
			height = new_pts[3] - new_pts[1]

			if _in_rule(width, height, cropped, occluded, truncated):
				x.append('{:.6f} {:.6f} {:.6f} {:.6f} {} {} {}\n'.format(new_pts[0], new_pts[1], new_pts[2], new_pts[3], int(cropped), int(occluded), int(truncated)))

	if x != [] and savefile:

		current_data[filename] = {}
		current_data[filename]['car_box2d'] = x

	return len(x)


################################



def _bbox_lane(filename):

	with open(os.path.join(json_dir, filename + '.json'), 'r') as f:
		data = json.load(f)

	x = []

	for obj in data['frames'][0]['objects']:
		if obj['category'] == 'area/drivable':
			for pt in obj['poly2d']:
				x.append(float(pt[0]))

			return min(x), max(x)

	x = []

	for obj in data['frames'][0]['objects']:
		if obj['category'] == 'area/alternative':
			for pt in obj['poly2d']:
				x.append(float(pt[0]))

			return min(x), max(x)


	return _try_bbox_alternative(filename)	


def _try_bbox_alternative(filename):

	global without_car

	h = 720.0
	sw_A = 0
	sw_B = 280
	sw_C = 560
	n_cars_A = _adjust_car_box2d(filename, sw_A, h, savefile=False)
	n_cars_B = _adjust_car_box2d(filename, sw_B, h, savefile=False)
	n_cars_C = _adjust_car_box2d(filename, sw_C, h, savefile=False)

	n_cars = 0
	sw = 0

	if n_cars_A > n_cars_B:
		sw = sw_A
		n_cars = n_cars_A
	else:
		sw = sw_B
		n_cars = n_cars_B

	if n_cars_C > n_cars:
		sw = sw_C
		n_cars = n_cars_C

	if n_cars == 0:
		without_car += 1
		print('ERROR ! there is no car: {} | {}'.format(filename, without_car))
		return None, None
	else:
		return sw, sw+h


def _square(img, minn, maxx):
	
	w, h = img.size
	lane = maxx - minn

	mid_lane = minn + (lane/2.0)
	sw = max(0, mid_lane - (h/2.0))

	if sw + h > w:
		sw = w - h

	sw = int(sw)

	cropped_img = img.crop((sw, 0, sw+h, h))
	return cropped_img, h, sw


def _shrink(cropped_img):

	return cropped_img.resize((side, side), resample=Image.NEAREST)



def square_and_shrink(images_folder, labels_folder):

	global current_data, json_dir

	def _try_to_crop(f, minn, maxx):

		if minn != None and maxx != None:
		
			img = Image.open(os.path.join(images_folder, f + '.jpg'))
			cropped_img, h, sw = _square(img, minn, maxx)
			n_cars = _adjust_car_box2d(f, sw, h)

			if n_cars > 0:
				resized_img = _shrink(cropped_img)
				current_data[f]['img'] = resized_img
			else:
				minn, maxx = _try_bbox_alternative(f)
				_try_to_crop(f, minn, maxx)


	filenames = os.listdir(images_folder)
	filenames = [f[:-4] for f in filenames]
	l = len(filenames)
	i = 0

	json_dir = labels_folder
	current_data = {}

	print('Processing the images and labels of folders {} and {}:'.format(images_folder, labels_folder))

	for f in tqdm(filenames):

		minn, maxx = _bbox_lane(f)
		_try_to_crop(f, minn, maxx)

		os.remove(os.path.join(images_folder, f + '.jpg'))
		current_data[f]['img'].save(os.path.join(images_folder, f + '.jpg'))

		os.remove(os.path.join(labels_folder, f + '.json'))
		with open(os.path.join(labels_folder, f + '.car_box2d'), 'w') as file:
			file.writelines(current_data[f]['car_box2d'])



img_train_day_dir_to = os.path.join('images', 'train', 'day')
img_train_night_dir_to = os.path.join('images', 'train', 'night')

img_test_day_dir_to = os.path.join('images', 'test', 'day')
img_test_night_dir_to = os.path.join('images', 'test', 'night')

lab_train_day_dir_to = os.path.join('labels', 'train', 'day')
lab_train_night_dir_to = os.path.join('labels', 'train', 'night')

lab_test_day_dir_to = os.path.join('labels', 'test', 'day')
lab_test_night_dir_to = os.path.join('labels', 'test', 'night')


square_and_shrink(img_train_day_dir_to, lab_train_day_dir_to)
square_and_shrink(img_train_night_dir_to, lab_train_night_dir_to)
square_and_shrink(img_test_day_dir_to, lab_test_day_dir_to)
square_and_shrink(img_test_night_dir_to, lab_test_night_dir_to)

