from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import cv2
import numpy as np

from pathlib import Path
from numpy import ndarray
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import torch
from typing import List
import os
from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path
from typing import Any, Union
import yaml
import pandas as pd
import tqdm
import pickle
from PIL import Image
import io
import base64
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.tag import Tag
from pydicom.uid import ExplicitVRLittleEndian
from django.conf import settings


from django.http import JsonResponse, Http404
from django.views.decorators.csrf import csrf_exempt

# from full_convmf_vn import ConvMF, Recommender
import numpy as np
MEASURE_DOT = "dot product aka. inner product"

def load_yaml(filepath: Union[str, Path]) -> Any:
    with open(filepath, "r") as f:
        content = yaml.full_load(f)
    return content

print('location: ', os.getcwd())

@dataclass
class Flags:
    # General
    debug: bool = True
    outdir: str = "results/det"

    # Data config
    imgdir_name: str = "vinbigdata-chest-xray-resized-png-256x256"
    split_mode: str = "all_train"  # all_train or valid20
    seed: int = 111
    train_data_type: str = "original"  # original or wbf
    use_class14: bool = False
    # Training config
    iter: int = 10000
    ims_per_batch: int = 2  # images per batch, this corresponds to "total batch size"
    num_workers: int = 4
    lr_scheduler_name: str = "WarmupMultiStepLR"  # WarmupMultiStepLR (default) or WarmupCosineLR
    base_lr: float = 0.00025
    roi_batch_size_per_image: int = 512
    eval_period: int = 10000
    aug_kwargs: Dict = field(default_factory=lambda: {})

    def update(self, param_dict: Dict) -> "Flags":
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self
    


def format_pred(labels: ndarray, boxes: ndarray, scores: ndarray) -> str:
    pred_strings = []
    for label, score, bbox in zip(labels, scores, boxes):
        xmin, ymin, xmax, ymax = bbox.astype(np.int64)
        pred_strings.append(f"{label} {score} {xmin} {ymin} {xmax} {ymax}")
    return " ".join(pred_strings)


def predict_batch(predictor: DefaultPredictor, im_list: List[ndarray]) -> List:
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        inputs_list = []
        for original_image in im_list:
            # Apply pre-processing to image.
            if predictor.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # Do not apply original augmentation, which is resize.
            # image = predictor.aug.get_transform(original_image).apply_image(original_image)
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            inputs_list.append(inputs)
        predictions = predictor.model(inputs_list)
        return predictions
    

thing_classes = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis"
]
category_name_to_id = {class_name: index for index, class_name in enumerate(thing_classes)}

def get_vinbigdata_dicts_test(
    imgdir: Path, test_meta: pd.DataFrame, use_cache: bool = True, debug: bool = True,
):
    debug_str = f"_debug{int(debug)}"
    cache_path = Path(".") / f"dataset_dicts_cache_test{debug_str}.pkl"
    if not use_cache or not cache_path.exists():
        print("Creating data...")
        # test_meta = pd.read_csv(imgdir / "test_meta.csv")
        if debug:
            test_meta = test_meta.iloc[:500]  # For debug....

        # Load 1 image to get image size.
        image_id = test_meta.loc[0, "image_id"]
        image_path = str(imgdir / "test" / f"{image_id}.png")
        image = cv2.imread(image_path)
        resized_height, resized_width, ch = image.shape
        print(f"image shape: {image.shape}")

        dataset_dicts = []
        for index, test_meta_row in tqdm(test_meta.iterrows(), total=len(test_meta)):
            record = {}

            image_id, height, width = test_meta_row.values
            filename = str(imgdir / "test" / f"{image_id}.png")
            record["file_name"] = filename
            # record["image_id"] = index
            record["image_id"] = image_id
            record["height"] = resized_height
            record["width"] = resized_width
            # objs = []
            # record["annotations"] = objs
            dataset_dicts.append(record)
        with open(cache_path, mode="wb") as f:
            pickle.dump(dataset_dicts, f)

    print(f"Load from cache {cache_path}")
    with open(cache_path, mode="rb") as f:
        dataset_dicts = pickle.load(f)
    return dataset_dicts


inputdir = Path("./..")
traineddir = inputdir / "vinbigdata-alb-aug-512-cos"
# traineddir = inputdir
print('traineddir: ',traineddir)


# flags = Flags()
flags: Flags = Flags().update(load_yaml(str(traineddir/"flags.yaml")))
print("flags", flags)
debug = flags.debug
# flags_dict = dataclasses.asdict(flags)
outdir = Path(flags.outdir)
os.makedirs(str(outdir), exist_ok=True)


datadir = inputdir / "vinbigdata-chest-xray-abnormalities-detection"
if flags.imgdir_name == "vinbigdata-chest-xray-resized-png-512x512":
    imgdir = inputdir/ "vinbigdata"
    print('size test 512')
else:
    imgdir = inputdir / flags.imgdir_name
    print('size test 256')
test_meta = pd.read_csv(inputdir / "vinbigdata-testmeta" / "test_meta.csv")


cfg = get_cfg()
original_output_dir = cfg.OUTPUT_DIR
cfg.OUTPUT_DIR = str(outdir)
print(f"cfg.OUTPUT_DIR {original_output_dir} -> {cfg.OUTPUT_DIR}")

cfg.MODEL.DEVICE = "cpu"

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("vinbigdata_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = flags.base_lr  # pick a good LR
cfg.SOLVER.MAX_ITER = flags.iter
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = flags.roi_batch_size_per_image
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

### --- Inference & Evaluation ---
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
# path to the model we just trained
cfg.MODEL.WEIGHTS = str(traineddir/"model_final.pth")
print("Original thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)  # 0.05
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
print("Changed  thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
predictor = DefaultPredictor(cfg)

if "vinbigdata_test" in DatasetCatalog:
    DatasetCatalog.remove("vinbigdata_test")

DatasetCatalog.register(
    "vinbigdata_test", lambda: get_vinbigdata_dicts_test(imgdir, test_meta, debug=debug)
)
MetadataCatalog.get("vinbigdata_test").set(thing_classes=thing_classes)
metadata = MetadataCatalog.get("vinbigdata_test")
# dataset_dicts = get_vinbigdata_dicts_test(imgdir, test_meta, debug=debug)

# if debug:
#     dataset_dicts = dataset_dicts[:100]

results_list = []
index = 0
batch_size = 4

def check_pixel_data_encapsulation(dataset):
    """Kiểm tra Pixel Data có được encapsulated hay không"""
    try:
        pixel_data_tag = Tag(0x7FE0, 0x0010)  # Tag của Pixel Data
        if pixel_data_tag in dataset and dataset.file_meta.TransferSyntaxUID.is_compressed:
            return True
        return False
    except AttributeError:
        return False

# Create your views here.
def home(request):
    return render(request, 'app/pages/home.html')

def services(request):
    context = {}
    context['service'] = ''
    context['result'] = ''
    if request.method == 'POST':
        if request.FILES.get('img') != None:
            uploaded_file = request.FILES['img']
            context['service'] = 'img'
        fileSystemStorage = FileSystemStorage()
        fileSystemStorage.save(uploaded_file.name, uploaded_file)
        context['url'] = fileSystemStorage.url(uploaded_file)
        file_path = '.' + context['url']

        filename_org = uploaded_file.name.split('.')[0]
        is_dicom = False

        try:
            # DICOM file check
            dicom_data = pydicom.dcmread(file_path)
            print("DICOM file detected.", dicom_data)


            org_BitsAllocated = dicom_data.BitsAllocated  
            org_BitStored = dicom_data.BitsStored
            org_HighBit = dicom_data.HighBit 
            org_PixelRepresentation = dicom_data.PixelRepresentation

            print('org_BitAllocated: ', org_BitsAllocated)
            print('org_BitStored: ', org_BitStored)
            print('org_HighBit: ', org_HighBit)
            print('org_PixelRepresentation: ', org_PixelRepresentation)

            # for elem in dicom_data:
            #     print(f"{elem.keyword} - {elem.value}/n")

            data = apply_voi_lut(dicom_data.pixel_array, dicom_data)
            if dicom_data.PhotometricInterpretation == "MONOCHROME1":
                data = np.amax(data) - data
            
            data = data - np.min(data)
            data = data / np.max(data)
            data = (data * 255).astype(np.uint8)

            im = Image.fromarray(data)
            im = im.resize((512, 512), Image.LANCZOS)

            im_array = np.array(im)

            # display in html
            img_byte_arr = io.BytesIO()
            im.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            encoded_img = base64.b64encode(img_byte_arr.read()).decode('utf-8')
            context['service'] = 'dicom'
            context['dicom_image'] = f"data:image/png;base64,{encoded_img}"

            # Convert DICOM to image
            # image = dicom_data.pixel_array
            image = cv2.cvtColor(im_array, cv2.COLOR_GRAY2RGB)  # Chuyển đổi sang RGB nếu cần
            is_dicom = True
        except pydicom.errors.InvalidDicomError:
            print("Not a DICOM file, assuming standard image format.")
            # Đọc ảnh thông thường nếu không phải DICOM
            try:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Unsupported file format.")
            except Exception as e:
                context['error'] = f"File format not supported: {str(e)}"
                return render(request, 'app/pages/services.html', context)

        # image = cv2.imread(img)
        single_prediction = predict_batch(predictor, [image])
        prediction = single_prediction[0]
        print("prediction: ", prediction)

        v = Visualizer(
            image[:, :, ::-1],
            metadata=MetadataCatalog.get("vinbigdata_test"),
            scale=1,
            instance_mode=ColorMode.IMAGE_BW
        )
        output = v.draw_instance_predictions(prediction["instances"].to("cpu"))

        # Chuyển đổi ảnh output thành base64 để hiển thị trong template
        output_image = output.get_image()[:, :, ::-1]

        pil_img = Image.fromarray(output_image)
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        encoded_img = base64.b64encode(img_byte_arr.read()).decode('utf-8')
        context['encoded_img'] = encoded_img


        # save file dicom
        if is_dicom:
            boxes = prediction["instances"].pred_boxes.tensor.cpu().numpy()  # Tọa độ bounding box
            classes = prediction["instances"].pred_classes.cpu().numpy()     # Lớp dự đoán
            scores = prediction["instances"].scores.cpu().numpy()            # Xác suất

            # Nếu resize ảnh, tính lại tọa độ bounding box
            scale_x = dicom_data.Columns / im_array.shape[1]
            scale_y = dicom_data.Rows / im_array.shape[0]
            boxes = boxes * [scale_x, scale_y, scale_x, scale_y]

            # Vẽ bounding box lên ảnh gốc
            image_with_boxes = data.copy()  # Dữ liệu pixel gốc từ DICOM
            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = map(int, box)
                label = f"{thing_classes[cls]} ({score:.2f})"
                
                # Vẽ bounding box
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 6)
                
                # Thêm nhãn
                cv2.putText(image_with_boxes, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6)

            # save to Result folder
            # result_dir = "./Result"
            # if not os.path.exists(result_dir):
            #     os.makedirs(result_dir)

            # # Lưu file DICOM đã chỉnh sửa
            # dicom_file_path = os.path.join(result_dir, "modified3.dicom")

            if dicom_data.file_meta.TransferSyntaxUID.is_compressed:
                # If the original was compressed, we need to change to uncompressed
                dicom_data.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            if dicom_data.BitsAllocated == 16:
                # Chuyển đổi ảnh về 16-bit
                image_with_boxes = image_with_boxes.astype(np.uint16)
                image_with_boxes = (image_with_boxes / 255.0 * (2**16 - 1)).astype(np.uint16)

                # Cập nhật PixelData
                dicom_data.PixelData = image_with_boxes.tobytes()
            else:
                # Nếu là ảnh 8-bit
                dicom_data.PixelData = image_with_boxes.tobytes()

            # Cập nhật Rows, Columns
            dicom_data.Rows, dicom_data.Columns = image_with_boxes.shape[:2]


            dicom_data.Rows, dicom_data.Columns = image_with_boxes.shape[:2]
            dicom_data.SamplesPerPixel = 1 if len(image_with_boxes.shape) == 2 else image_with_boxes.shape[2]
            dicom_data.PhotometricInterpretation = "MONOCHROME2"  # Hoặc RGB nếu ảnh là màu

            # dicom_data.BitsAllocated = 8  # hoặc 16 tùy thuộc vào định dạng ảnh
            # dicom_data.BitsStored = 8  # hoặc 16
            # dicom_data.HighBit = 7 if dicom_data.BitsStored == 8 else 15
            # dicom_data.PixelRepresentation = 0  # 0: unsigned int, 1: signed int

            # dicom_data.BitsAllocated = 16  # hoặc 16 tùy thuộc vào định dạng ảnh
            # dicom_data.BitsStored = 14  # hoặc 16
            # dicom_data.HighBit = 13
            # dicom_data.PixelRepresentation = 0  # 0: unsigned int, 1: signed int

            # Giữ nguyên các thông số từ file ban đầu
            dicom_data.BitsAllocated = dicom_data.BitsAllocated  # Giữ nguyên
            dicom_data.BitsStored = dicom_data.BitsStored  # Giữ nguyên
            dicom_data.HighBit = dicom_data.HighBit  # Giữ nguyên
            dicom_data.PixelRepresentation = dicom_data.PixelRepresentation  # Giữ nguyên (0: unsigned, 1: signed)

            result_dir = os.path.join(settings.MEDIA_ROOT, 'dicom_results')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
                
            filename = filename_org + "_RS.dicom"
            dicom_file_path = os.path.join(result_dir, filename)
            dicom_data.save_as(dicom_file_path)
            
            print(f"File saved at: {dicom_file_path}")  # Add this debug print
            context['dicom_file'] = filename
    return render(request, 'app/pages/services.html', context)

def about(request):
    return render(request, 'app/pages/about.html')

def contact(request):
    return render(request, 'app/pages/contact.html')


from django.http import FileResponse
import os

def download_dicom(request, filename):
    """View to handle DICOM file downloads"""
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, 'dicom_results', filename)
        if os.path.exists(file_path):
            response = FileResponse(open(file_path, 'rb'))
            response['Content-Type'] = 'application/dicom'
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
        else:
            raise Http404("File not found")
    except Exception as e:
        raise Http404(f"Error accessing file: {str(e)}")


