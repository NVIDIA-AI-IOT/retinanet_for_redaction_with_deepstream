"""
Tools for manipulating coco annotaitons
"""

import os
import json
from PIL import Image
import random
import shutil

def resize(img_folder, annotations, resize_factor, output_img_folder, output_annotations):
    """
    Resize images to (original size * resize_factor)
    :param img_folder: Folder containing original images
    :param annotations: File containing COCO style annotations
    :param resize_factor: factor to increase each dim size by. 0.25 = shrink by 4 x
    :param output_img_folder: Folder that will contain the new images
    :param output_annotations: File that will contain the new annotations.
    :return:
    """
    # Check all files and directories exist
    assert os.path.isdir(img_folder), "Directory %s does not exist" % img_folder
    assert os.path.isdir(output_img_folder), "Directory %s does not exist" % output_img_folder
    assert os.path.isdir(os.path.split(output_annotations)[0]), "Directory %s does not exist" % os.path.split(output_annotations)[0]
    assert os.path.isfile(annotations), "File %s does not exist" % annotations

    # Read in annotations
    print("Reading annotaitons from", annotations)
    with open(annotations) as f:
        anns = json.load(f)
    if not anns:
        raise IOError("The annotation file is empty.")

    new_images = []
    old_images = anns['images']
    # Work through each image in annotations, resizing height & width attributes and resizing and copying image.
    for img in old_images:
        old_filepath = os.path.join(img_folder, img['file_name'])
        new_filepath = os.path.join(output_img_folder, img['file_name'])
        new_w = int(resize_factor * img['width'])
        new_h = int(resize_factor * img['height'])
        img['width'] = new_w
        img['height'] = new_h
        new_images.append(img)

        # Now resize
        try:
            image = Image.open(old_filepath).convert("RGB")
        except FileNotFoundError:
            print("Image not found:", old_filepath)
            continue
        except OSError:
            print("Image damaged:", old_filepath)
            continue
        new_image = image.resize((new_w, new_h), Image.BILINEAR)
        new_image.save(new_filepath, quality=95)

        # print("DEBUG: Image size requested:", new_w, new_h)
        # print("DEBUG: new_image.size", new_image.size)
        # print("DEBUG: new_image location", new_filepath)
        #
        # raise NotImplementedError("TEST RESIZE)")

    anns['images'] = new_images

    # Work through annotations, resizing xmin, ymin, w, h.
    old_anns = anns['annotations']
    new_anns = []
    for ann in old_anns:
        [xmin, ymin, w, h] = ann['bbox']
        xmin = int(xmin * resize_factor)
        ymin = int(ymin * resize_factor)
        w = int(w * resize_factor)
        h = int(h * resize_factor)
        xmax = xmin + w
        ymax = ymin + h
        ann['bbox'] = [xmin,  ymin, w, h]
        ann['area'] = w * h
        ann['seg'] = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
        new_anns.append(ann)
    anns['annotations'] = new_anns
    # Save out new annotations.

    print("All images resized and copied.")
    with open(output_annotations, 'w') as outfile:
        json.dump(anns, outfile)


def split_dataset(input_annotations, frac_split_a, a_output_path, b_output_path):
    """
    Split the dataset into two fractions, a and b.
    :param input_annotations:
    :param frac_split_a: 0.8 = 80% of data goes to a, 20% to b
    :param a_output_path:
    :param b_output_path:
    :return:
    """
    # Check all files and directories exist
    assert os.path.isdir(os.path.split(a_output_path)[0]), "Directory %s does not exist" % os.path.split(a_output_path)[0]
    assert os.path.isdir(os.path.split(b_output_path)[0]), "Directory %s does not exist" % os.path.split(b_output_path)[0]
    assert os.path.isfile(input_annotations), "File %s does not exist" % input_annotations
    assert 0 < frac_split_a < 1, "frac_split must be between 0 and 1"

    # Read in annotations
    print("Reading annotaitons from", input_annotations)
    with open(input_annotations) as f:
        input_anns = json.load(f)
    if not input_anns:
        raise IOError("The annotation file is empty.")

    # Loop through images, assigning each to either 'a' or 'b'.
    image_split_dict = {}
    for im in input_anns['images']:
        if random.random() < frac_split_a:
            image_split_dict[im['id']] = 'a'
        else:
            image_split_dict[im['id']] = 'b'

    # Now create two outputs and assign image and annotations to each.
    output_a = {'info':
                  input_anns['info'] + ' Split ' + str(frac_split_a),
              'licenses': input_anns['licenses'],
              'images': [],
              'annotations': [],
              'categories': input_anns['categories']}  # Prepare output
    output_b = {'info':
                  input_anns['info'] + ' Split ' + str(1 - frac_split_a),
              'licenses': input_anns['licenses'],
              'images': [],
              'annotations': [],
              'categories': input_anns['categories']}  # Prepare output

    output_a_raw_images = []
    output_b_raw_images = []

    for im in input_anns['images']:
        if image_split_dict[im['id']] == 'a':
            output_a_raw_images.append(im)
        elif image_split_dict[im['id']] == 'b':
            output_b_raw_images.append(im)
        else:
            raise SyntaxError('im not assigned to a nor b.', im)

    output_a_raw_anns = []
    output_b_raw_anns = []

    for ann in input_anns['annotations']:
        if image_split_dict[ann['image_id']] == 'a':
            output_a_raw_anns.append(ann)
        elif image_split_dict[ann['image_id']] == 'b':
            output_b_raw_anns.append(ann)
        else:
            raise SyntaxError('ann not assigned to a nor b.', ann)

    # Now renumber the annotations and images in each
    output_a_images = []
    a_oldimgid2newimgid = {} # to use to convert annotations
    for indx, aa in enumerate(output_a_raw_images):
        a_oldimgid2newimgid[aa['id']] = indx
        aa['id'] = indx
        output_a_images.append(aa)
    output_b_images = []
    b_oldimgid2newimgid = {} # to use to convert annotations
    for indx, bb in enumerate(output_b_raw_images):
        b_oldimgid2newimgid[bb['id']] = indx
        bb['id'] = indx
        output_b_images.append(bb)

    output_a_annotations = []
    for indx, aa in enumerate(output_a_raw_anns):
        new_image_id = a_oldimgid2newimgid[aa['image_id']]
        aa['image_id'] = new_image_id
        aa['id'] = indx
        output_a_annotations.append(aa)

    output_b_annotations = []
    for indx, bb in enumerate(output_b_raw_anns):
        new_image_id = b_oldimgid2newimgid[bb['image_id']]
        bb['image_id'] = new_image_id
        bb['id'] = indx
        output_b_annotations.append(bb)

    output_a['images'] = output_a_images
    output_a['annotations'] = output_a_annotations

    output_b['images'] = output_b_images
    output_b['annotations'] = output_b_annotations

    # Write some info
    print("Split A contains %i images and %i annotations." % (len(output_a['images']), len(output_a['annotations'])))
    print("Split B contains %i images and %i annotations." % (len(output_b['images']), len(output_b['annotations'])))

    # Write each out
    with open(a_output_path, 'w') as outfile:
        json.dump(output_a, outfile)
    with open(b_output_path, 'w') as outfile:
        json.dump(output_b, outfile)


def copy_images(all_img_dir, new_img_dir, ann_file):
    """
    Copy all images mentioned in ann_file from all_img_dir to new_img_dir
    """
    assert os.path.isdir(all_img_dir), "Directory %s does not exist" % all_img_dir
    assert os.path.isdir(new_img_dir), "Directory %s does not exist" % new_img_dir
    assert os.path.isfile(ann_file), "File %s does not exist" % ann_file

    print("Reading annotaitons from", ann_file)
    with open(ann_file) as f:
        input_anns = json.load(f)
    if not input_anns:
        raise IOError("The annotation file is empty.")

    # Get list of images
    img_list = [im['file_name'] for im in input_anns['images']]

    # Work through list
    for im in img_list:
        old_path = os.path.join(all_img_dir, im)
        new_path = os.path.join(new_img_dir, im)
        shutil.copy(old_path, new_path)

def read_json(coco_annotation, verbose=False):
    if verbose:
        print("Reading annotaitons from", coco_annotation)
    with open(coco_annotation) as f:
        anns = json.load(f)
    if not anns:
        raise IOError("The annotation file is empty.")
    return anns

def write_json(data, filepath):
    """Write JSON file"""
    dir_ = os.path.split(filepath)[0]
    assert os.path.isdir(dir_), "Directory %s does not exist" % dir_

    with open(filepath, 'w') as outfile:
        json.dump(data, outfile)

def get_filename2imgid(annfile, verbose=False):
    anns = read_json(annfile, verbose=verbose)
    filename2imgid = {}
    for img in anns['images']:
        filename2imgid[img['file_name']] = img['id']
    return filename2imgid

def get_imgid2anns(annfile, verbose=False):
    anns = read_json(annfile, verbose=verbose)
    imgid2anns = {}
    for ann in anns['annotations']:
        imgid = ann['image_id']
        if imgid not in imgid2anns:
            imgid2anns[imgid] = []
        imgid2anns[imgid].append(ann)
    return imgid2anns

def get_imgid2img(annfile, verbose=False):
    anns = read_json(annfile, verbose=verbose)
    imgid2img = {}
    for img in anns['images']:
        imgid2img[img['id']] = img
    return imgid2img



