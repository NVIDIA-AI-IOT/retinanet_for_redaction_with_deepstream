import os, csv, json, shutil
from data_tools.coco_tools import read_json
from PIL import Image


def reduce_data(oidata, catmid2name, keep_classes=[]):
    """
    Reduce the amount of data by only keeping images that are in the classes we want.
    :param oidata: oidata, as outputted by parse_open_images
    :param catmid2name: catid2name dict, as produced by read_catMIDtoname
    :param keep_classes: List of classes to be kept.
    :return:
    """
    print(" Reducing the dataset. Initial dataset has length", len(oidata))
    # First build a dictionary of imageID:[classnames]
    imageid2classmid = {}
    for dd in oidata:
        imageid = dd['ImageID']
        if imageid not in imageid2classmid:
            imageid2classmid[imageid] = [dd['LabelName']]
        else:
            imageid2classmid[imageid].append(dd['LabelName'])

    # Work out which images we are including.
    imageid2include = {} # dict to store True if this imageid is included.

    for imgid, classmids in imageid2classmid.items():
        imageid2include[imgid] = False # Assume we don't include this.
        for mid in classmids:
            this_name = catmid2name[mid]
            if this_name in keep_classes:
                imageid2include[imgid] = True

    # Now work through list, appending if ImageID has imageid2include[imageid] = True
    returned_data = []
    for dd in oidata:
        imageid = dd['ImageID']
        if imageid2include[imageid]:
            returned_data.append(dd)

    print(" Reducing the dataset. Final dataset has length", len(returned_data))
    return returned_data

def openimages2coco(oidata, catmid2name, img_dir, desc="", output_class_ids=None,
                    max_size=None, min_ann_size=None, min_ratio=0.0, min_width_for_ratio=400):
    """
    Converts open images annotations into COCO format
    :param raw: list of data items, as produced by parse_open_images
    :return: COCO style dict
    """
    output = {'info':
                  "Annotations produced from OpenImages. %s" % desc,
              'licenses': [],
              'images': [],
              'annotations': [],
              'categories': []} # Prepare output

    # Get categories in this dataset
    all_cats = []
    for dd in oidata:
        if dd['LabelName'] not in all_cats:
            all_cats.append(dd['LabelName'])
    categories = []
    for mid in all_cats:
        cat_name = catmid2name[mid]
        if cat_name in output_class_ids:
            categories.append({"id": output_class_ids[cat_name], "name": cat_name, "supercategory": 'object'})
    output['categories'] = categories

    # Get images
    image_filename_to_id = {} # To store found images.
    current_img_index = 0 #To incrementally add image IDs.
    imgid2wh = {} # To store width and height
    intermediate_images = [] # To store as if output
    for dd in oidata:
        filename = dd['ImageID'] + '.jpg'
        if filename not in image_filename_to_id:
            img_entry = _oidata_entry_to_image_dict(filename, current_img_index, img_dir)
            image_filename_to_id[filename] = current_img_index
            imgid2wh[current_img_index] = (img_entry['width'], img_entry['height'])
            intermediate_images.append(img_entry)
            current_img_index += 1

    # Get annotations
    ann_id = 1
    imgid2_has_new_ann = {} # Use this to make sure that our images have valid annotations
    new_anns_raw = [] # list of candidate annotations
    for dd in oidata:
        filename = dd['ImageID'] + '.jpg'
        imgid = image_filename_to_id[filename]
        cat_name = catmid2name[dd['LabelName']]
        if cat_name in output_class_ids:
            catid = output_class_ids[cat_name]
            w, h = imgid2wh[imgid]
            bbox, area, seg = _ann2bbox(dd, w, h)
            ann_entry = {'id': ann_id, 'image_id': imgid, 'category_id': catid,
                         'segmentation': seg,
                         'area': area,
                         'bbox': bbox,
                         'iscrowd': 0}
            # Check if we want to include this annotation
            include_this_annotation = True
            x, y, ann_w, ann_h = bbox
            if max_size:
                maxdim = max(w, h)
                ann_w = ann_w * (max_size / float(maxdim))
                ann_h = ann_h * (max_size / float(maxdim))
            if min_ann_size is not None:
                if ann_w < min_ann_size[0]:
                    include_this_annotation = False
                if ann_h < min_ann_size[1]:
                    include_this_annotation = False

            # Now check whether this annotation exceeds the ratio requriements, if any.
            if min_ratio > 0:
                try:
                    ratio = float(w) / float(h)
                except ZeroDivisionError:
                    include_this_annotation = False
                else:
                    if ratio >= min_ratio and w >= min_width_for_ratio:
                        include_this_annotation = False

            if include_this_annotation:
                new_anns_raw.append(ann_entry)
                imgid2_has_new_ann[imgid] = True
                ann_id += 1

    # Now we must review all of the images and only keep those where imgid2_has_new_ann[imgid] = True

    new_imgs_raw = []
    for img in intermediate_images:
        if img['id'] in imgid2_has_new_ann:
            new_imgs_raw.append(img)

    # Now we assign new image_ids to the images, mapping old to new
    old_img2new_img = {}
    new_imgs = []
    for indx, img in enumerate(new_imgs_raw):
        old_img2new_img[img['id']] = indx + 1
        img['id'] = indx + 1
        new_imgs.append(img)

    output['images'] = new_imgs

    # Now we assing new ann_ids to the annotations, also updating the image ID
    new_anns = []
    for indx, ann in enumerate(new_anns_raw):
        ann['id'] = indx + 1
        ann['image_id'] = old_img2new_img[ann['image_id']]
        new_anns.append(ann)

    output['annotations'] = new_anns
    return output

def read_catMIDtoname(csv_file):
    catmid2name = {}

    assert os.path.isfile(csv_file), "File %s does not exist." % csv_file

    rows_read = 0
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            mid = row[0]
            name = row[1]
            catmid2name[mid] = name
            rows_read += 1
    print(" Read", rows_read, "rows from category csv", csv_file)
    return catmid2name

def parse_open_images(annotation_csv):
    """
    Parse open images and produce a list of annotations.
    :param annotation_csv:
    :return:
    """
    annotations = []

    assert os.path.isfile(annotation_csv), "File %s does not exist." % annotation_csv
    expected_header = ['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']

    rows_read = 0
    with open(annotation_csv) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for ii, hh in enumerate(header):
            assert hh == expected_header[ii], "File header is not as expected."
        for row in reader:
            ann = parse_open_images_row(row, header)
            annotations.append(ann)
            rows_read += 1
            # if rows_read > 10:
            #     print("DEBUG: Only reading 11 rows.")
            #     break
    print(" Read", rows_read, "rows from annotation csv", annotation_csv)
    return annotations

def parse_open_images_row(row, header):
    """Parse open images row, returning a dict
    Format of dict (str unless otherwise specified)
    ImageID: Image ID of the box.
    Source: Indicateds how the box was made.
        xclick are manually drawn boxes using the method presented in [1].
        activemil are boxes produced using an enhanced version of the method [2]. These are human verified to be accurate at IoU>0.7.
    LabelName: MID of the object class
    Confidence: Always 1 (here True)
    XMin, XMax, YMin, YMax: coordinates of the box, in normalized image coordinates. (FLOAT)
        XMin is in [0,1], where 0 is the leftmost pixel, and 1 is the rightmost pixel in the image.
        Y coordinates go from the top pixel (0) to the bottom pixel (1).
    For each of them, value 1 indicates present, 0 not present, and -1 unknown. (INT)
        IsOccluded: Indicates that the object is occluded by another object in the image.
        IsTruncated: Indicates that the object extends beyond the boundary of the image.
        IsGroupOf: Indicates that the box spans a group of objects (e.g., a bed of flowers or a crowd of people). We asked annotators to use this tag for cases with more than 5 instances which are heavily occluding each other and are physically touching.
        IsDepiction: Indicates that the object is a depiction (e.g., a cartoon or drawing of the object, not a real physical instance).
        IsInside: Indicates a picture taken from the inside of the object (e.g., a car interior or inside of a building).

    """
    ann = {}
    for ii, hh in enumerate(header):
        if hh in ['XMin', 'XMax', 'YMin', 'YMax']:
            ann[hh] = float(row[ii])
        elif hh in ['Confidence', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']:
            ann[hh] = int(row[ii])
        else: # str
            ann[hh] = row[ii]
    return ann

def copy_images(json_file, original_image_dirs, new_image_dir):
    """Copy files from original_image_dirs to new_iamge_dirs"""
    if type(original_image_dirs) is not list:
        original_image_dirs = [original_image_dirs]

    # Open JSON file and get list of images
    annotations = read_json(json_file, verbose=False)
    image_filenames = [ann['file_name'] for ann in annotations['images']]

    for img in image_filenames:
        for img_d in original_image_dirs:
            orig = os.path.join(img_d, img)
            if not os.path.isfile(orig):
                continue
            new = os.path.join(new_image_dir, img)
            # Copy
            shutil.copy(orig, new)
    print("All %i images in %s copied to %s" % (len(image_filenames), json_file, new_image_dir))


def _oidata_entry_to_image_dict(filename, indx, img_dir):
    width, height = _get_img_width_height(filename, img_dir)
    return {'id': indx, 'width': width, 'height': height, 'file_name': filename,
            'license': None, 'flickr_url': None, 'coco_url': None, 'date_captured': None}

def _get_img_width_height(filename, img_dir):
    # Modified to deal with img_dir as a list.
    if not type(img_dir) == list:
        img_dir = [img_dir]
    for img_d in img_dir:
        filepath = os.path.join(img_d, filename)
        try:
            image = Image.open(filepath).convert("RGB")
        except FileNotFoundError:
            pass
        else:
            return image.size
    raise FileNotFoundError("Image %s not found in any of img_dir" % filename)

def _ann2bbox(dd, img_width, img_height):
    xmin = dd['XMin'] * img_width
    xmax = dd['XMax'] * img_width
    ymin = dd['YMin'] * img_height
    ymax = dd['YMax'] * img_height
    seg = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
    w = xmax - xmin
    h = ymax - ymin
    bbox = [xmin, ymin, w, h]
    return bbox, w * h, seg

