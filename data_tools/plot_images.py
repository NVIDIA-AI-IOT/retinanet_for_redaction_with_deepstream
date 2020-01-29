from PIL import Image, ImageDraw, ImageFont
import os
from data_tools.coco_tools import read_json

def draw_boxes(image_dir, output_dir, anns):
    """
    Plot GT boxes
    """

    # Read annotations
    annotations=read_json(anns)

    # Get info we need
    catid2name = {}
    for cat in annotations['categories']:
        catid2name[cat['id']] = cat['name']


    imageid2filename = {}
    for ann in annotations['images']:
        imageid2filename[ann['id']] = ann['file_name']

    imageid2annboxes = {} # [(bbox, catid)]
    for ann in annotations['annotations']:
        this_entry = (ann['bbox'], ann['category_id'])
        if ann['image_id'] not in imageid2annboxes:
            imageid2annboxes[ann['image_id']] = [this_entry]
        else:
            imageid2annboxes[ann['image_id']].append(this_entry)

    # Now work through image IDs (imageid2filename.keys()) and create images.
    for done, imageid in enumerate(imageid2filename):
        filepath = os.path.join(image_dir, imageid2filename[imageid])
        image = Image.open(filepath).convert("RGB")
        # image = Image.open(filepath)
        draw = ImageDraw.Draw(image)
        # Add GT bounding boxes.
        anns = []
        try:
            anns = imageid2annboxes[imageid]
        except KeyError:
            pass
        for bbox, catid in anns:
            [xmin, ymin, w, h] = bbox
            xmax = int(round(xmin + w))
            ymax = int(round(ymin + h))
            draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)],
                      fill=(53, 111, 19), width=4)
            draw.text((xmin + 3, ymin - 18), catid2name[catid], (53, 111, 19))


        filename_root, filename_ext = os.path.splitext(imageid2filename[imageid])
        output_path = os.path.join(output_dir, filename_root + "_detections" + filename_ext)
        image.save(output_path)
        if (done + 1) % 25 == 0 and done > 0:
            print("  Saved {} of {} images.".format(done + 1, len(imageid2filename)),)
    print("  Saved {} of {} images.".format(done + 1, len(imageid2filename)), )


