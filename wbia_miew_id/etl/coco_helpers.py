import json
from tqdm.auto import tqdm
import cv2

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def write_json(data, out_path):
    json_object = json.dumps(data, indent=4)
    with open(out_path, "w") as outfile:
        outfile.write(json_object)
        
def export_annos(dfa, dfi, out_path):
    """ Used when two dataframes are passed in, one for annotations and one for images in the coco format. """
    print('out_path', out_path)
    print('shapes: ', dfa.shape, dfi.shape)
    annos_list = dfa.to_dict(orient='records')
    images_list = dfi.to_dict(orient='records')

    data = {
        'info':{},
        'licenses':[],
        'images':images_list,
        'annotations':annos_list,
        'parts':[]
           }
    write_json(data, out_path)

def get_image_dimensions(filename):
    img = cv2.imread(filename)
    h, w, c = img.shape
    return h, w

def convert_coco(df, images_dir):
    """
     Used to convert a single dataframe to coco format. 
    This is a minimalistic script with all non-critical fields are set to default values. 
    Bounding box coordinates are assumed to be same as full image.
    Assumes one annotation per image.
    """
    data = {
        'info':{},
        'licenses':[],
        'images':[],
        'annotations':[],
        'parts':[]
           }

    for i, row in tqdm(df.iterrows(), total=len(df)):

        file_name = row['Image']

        image_path = f"{images_dir}/{file_name}"
        image_h, image_w =  get_image_dimensions(image_path)

        image = {'license': 1,
             'file_name': file_name,
             'photographer': '',
             'coco_url': None,
             'height': image_h,
             'width': image_w,
             'date_captured': 'NA',
             'gps_lat_captured': '-1.000000',
             'gps_lon_captured': '-1.000000',
             'flickr_url': None,
             'id': i
            }

        ## same as picture - no bbox
        name = row['Id']
        viewpoint = row['Viewpoint']
        x, y, w, h = 0, 0, image_w, image_h
        

        annot = {'bbox': [x, y, w, h],
                  'theta': 'none',
                  'viewpoint': viewpoint,
                  'segmentation': [[y, x, y+h, x, y+h, x+w, y, x+w, y, x]],
                  'segmentation_bbox': [x, y, w, h],
                  'area': h*w,
                  'iscrowd': 0,
                  'id': i,
                  'image_id': i,
                  'category_id': 0,
                  'individual_ids': [],
                  'isinterest': 0,
                  'name': name
                }
        data['images'].append(image)
        data['annotations'].append(annot)
        

    return data