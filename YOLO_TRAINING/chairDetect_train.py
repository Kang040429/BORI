import os
import requests
import shutil
import random
import zipfile
import urllib.request
import argparse
from datetime import datetime
from tqdm import tqdm
from pycocotools.coco import COCO
from ultralytics import YOLO


# ==========================================
# 실행 인자 설정 (기본값 있음 → 그냥 실행해도 됨)
# 사용 예시:
#   python chairDetect_train.py
#   python chairDetect_train.py --epochs 30 --batch 16 --device 0
# ==========================================
parser = argparse.ArgumentParser(description='BORI 의자 YOLO 학습')
parser.add_argument('--epochs',    type=int, default=10,   help='학습 epoch 수 (기본값: 10)')
parser.add_argument('--batch',     type=int, default=16,   help='배치 크기 (기본값: 16)')
parser.add_argument('--val_count', type=int, default=1000, help='val 이미지 수 (기본값: 1000)')
parser.add_argument('--device',    type=str, default='0',  help='디바이스: 0=GPU, cpu=CPU (기본값: 0)')
args, _ = parser.parse_known_args()  # Colab/Kaggle 호환용

EPOCHS     = args.epochs
BATCH_SIZE = args.batch
VAL_COUNT  = args.val_count
DEVICE     = args.device if args.device == 'cpu' else int(args.device)

print("=" * 40)
print(f"  epochs    : {EPOCHS}")
print(f"  batch     : {BATCH_SIZE}")
print(f"  val_count : {VAL_COUNT}")
print(f"  device    : {DEVICE}")
print("=" * 40)


# ==========================================
# 환경 자동 감지
# ==========================================
def get_env():
    if os.path.exists('/kaggle/working'):
        return 'kaggle'
    elif os.path.exists('/content'):
        return 'colab'
    else:
        return 'local'

ENV = get_env()
print(f"[환경] {ENV}")

# ==========================================
# 환경별 경로 설정
# ==========================================
if ENV == 'kaggle':
    BASE_PATH        = '/kaggle/working'
    COCO_IMAGE_PATH  = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017'
    ANNOTATION_PATH  = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/instances_train2017.json'
    USE_LOCAL_IMAGES = True

elif ENV == 'colab':
    BASE_PATH        = '/content'
    COCO_IMAGE_PATH  = None
    ANNOTATION_PATH  = f'{BASE_PATH}/annotations/instances_train2017.json'
    USE_LOCAL_IMAGES = False

else:  # 젯슨, 로컬
    BASE_PATH        = os.path.dirname(os.path.abspath(__file__))
    COCO_IMAGE_PATH  = None
    ANNOTATION_PATH  = f'{BASE_PATH}/annotations/instances_train2017.json'
    USE_LOCAL_IMAGES = False

DATASET_PATH = f'{BASE_PATH}/dataset'
YAML_PATH    = f'{BASE_PATH}/chair_coco.yaml'
IMG_SIZE     = 640


# ==========================================
# 1. Annotation 다운로드 (Colab/로컬만)
# ==========================================
if not USE_LOCAL_IMAGES:
    if not os.path.exists(ANNOTATION_PATH):
        print("[1] COCO annotation 다운로드 중...")
        urllib.request.urlretrieve(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            f"{BASE_PATH}/annotations_trainval2017.zip"
        )
        with zipfile.ZipFile(f"{BASE_PATH}/annotations_trainval2017.zip", 'r') as z:
            z.extractall(BASE_PATH)
        print("[1] annotation 완료!")
    else:
        print("[1] annotation 이미 존재, 스킵")


# ==========================================
# 2. COCO 이미지 다운로드 + YOLO 변환
# ==========================================
print("[2] COCO chair 데이터 준비 중...")

coco = COCO(ANNOTATION_PATH)
chair_cat_id = coco.getCatIds(catNms=['chair'])[0]
img_ids = coco.getImgIds(catIds=[chair_cat_id])
print(f"    chair 이미지 수: {len(img_ids)}")

os.makedirs(f'{DATASET_PATH}/images/train', exist_ok=True)
os.makedirs(f'{DATASET_PATH}/labels/train', exist_ok=True)

for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    ann_ids  = coco.getAnnIds(imgIds=img_id, catIds=[chair_cat_id], iscrowd=False)
    anns     = coco.loadAnns(ann_ids)
    if not anns:
        continue

    img_dst = f"{DATASET_PATH}/images/train/{img_info['file_name']}"
    if not os.path.exists(img_dst):
        if USE_LOCAL_IMAGES:
            src = f"{COCO_IMAGE_PATH}/{img_info['file_name']}"
            shutil.copy(src, img_dst)
        else:
            r = requests.get(img_info['coco_url'], timeout=10)
            with open(img_dst, 'wb') as f:
                f.write(r.content)

    W, H = img_info['width'], img_info['height']
    lbl_path = f"{DATASET_PATH}/labels/train/{img_info['file_name'].replace('.jpg', '.txt')}"
    with open(lbl_path, 'w') as f:
        for ann in anns:
            x, y, w, h = ann['bbox']
            f.write(f"0 {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}\n")

print("[2] 완료!")


# ==========================================
# 2.7. Roboflow 라벨 데이터 추가
#      GitHub에서 chair.v1i.yolov8.zip 다운로드 후 병합
# ==========================================
GITHUB_ROBOFLOW_URL = 'https://raw.githubusercontent.com/Kang040429/BORI/main/YOLO_TRAINING/chair.v1i.yolov8.zip'

print("[2.7] Roboflow 라벨 데이터 다운로드 중...")
rf_zip = f'{BASE_PATH}/roboflow_chairs.zip'
ret = os.system(f'wget -q -O {rf_zip} "{GITHUB_ROBOFLOW_URL}"')

if ret == 0 and os.path.exists(rf_zip) and os.path.getsize(rf_zip) > 1000:
    import glob as _glob2
    rf_extract = f'{BASE_PATH}/roboflow_extract'
    os.makedirs(rf_extract, exist_ok=True)
    with zipfile.ZipFile(rf_zip, 'r') as z:
        z.extractall(rf_extract)

    rf_imgs  = _glob2.glob(f'{rf_extract}/**/*.jpg', recursive=True)
    rf_imgs += _glob2.glob(f'{rf_extract}/**/*.png', recursive=True)
    rf_imgs  = [p for p in rf_imgs if 'valid' not in p and 'test' not in p]
    rf_lbls  = _glob2.glob(f'{rf_extract}/**/*.txt', recursive=True)
    rf_lbls  = [p for p in rf_lbls if 'valid' not in p and 'test' not in p and 'data.yaml' not in p]

    for src in rf_imgs:
        dst = f"{DATASET_PATH}/images/train/rf_{os.path.basename(src)}"
        shutil.copy(src, dst)
    for src in rf_lbls:
        dst = f"{DATASET_PATH}/labels/train/rf_{os.path.basename(src)}"
        shutil.copy(src, dst)

    print(f"    Roboflow 이미지 {len(rf_imgs)}장, 라벨 {len(rf_lbls)}개 추가 완료")
else:
    print("    [주의] Roboflow zip 다운로드 실패 → 스킵")


# ==========================================
# 2.8. 오토라벨 영상 데이터 추가
#      GitHub에서 autolabel_vid3_part1~4.zip 다운로드 후 병합
# ==========================================
GITHUB_AUTOLABEL_BASE = 'https://media.githubusercontent.com/media/Kang040429/BORI/main/YOLO_TRAINING/autolabel_vid3'

print("[2.8] 오토라벨 데이터 다운로드 중...")
import glob as _glob3

al_total_imgs, al_total_lbls = 0, 0

for part in range(1, 5):
    al_zip = f'{BASE_PATH}/autolabel_vid3_part{part}.zip'
    url    = f'{GITHUB_AUTOLABEL_BASE}/autolabel_vid3_part{part}.zip'
    ret    = os.system(f'wget -q -O {al_zip} "{url}"')

    if ret == 0 and os.path.exists(al_zip) and os.path.getsize(al_zip) > 1000:
        al_extract = f'{BASE_PATH}/autolabel_vid3_part{part}'
        os.makedirs(al_extract, exist_ok=True)
        with zipfile.ZipFile(al_zip, 'r') as z:
            z.extractall(al_extract)

        al_imgs  = _glob3.glob(f'{al_extract}/**/*.jpg', recursive=True)
        al_imgs += _glob3.glob(f'{al_extract}/**/*.png', recursive=True)
        al_lbls  = _glob3.glob(f'{al_extract}/**/*.txt', recursive=True)

        for src in al_imgs:
            dst = f"{DATASET_PATH}/images/train/al_{os.path.basename(src)}"
            shutil.copy(src, dst)
        for src in al_lbls:
            dst = f"{DATASET_PATH}/labels/train/al_{os.path.basename(src)}"
            shutil.copy(src, dst)

        al_total_imgs += len(al_imgs)
        al_total_lbls += len(al_lbls)
        print(f"    part{part} 완료 ({len(al_imgs)}장)")
    else:
        print(f"    [주의] part{part} 다운로드 실패 → 스킵")

print(f"    오토라벨 총 {al_total_imgs}장, 라벨 {al_total_lbls}개 추가 완료")


# ==========================================
# 3. Val 분리
# ==========================================
print("[3] val 분리 중...")

os.makedirs(f'{DATASET_PATH}/images/val', exist_ok=True)
os.makedirs(f'{DATASET_PATH}/labels/val', exist_ok=True)

imgs = os.listdir(f'{DATASET_PATH}/images/train')
random.shuffle(imgs)
val_imgs = imgs[:VAL_COUNT]

for f in val_imgs:
    shutil.move(f'{DATASET_PATH}/images/train/{f}', f'{DATASET_PATH}/images/val/{f}')
    lbl = f.replace('.jpg', '.txt')
    if os.path.exists(f'{DATASET_PATH}/labels/train/{lbl}'):
        shutil.move(f'{DATASET_PATH}/labels/train/{lbl}', f'{DATASET_PATH}/labels/val/{lbl}')

print(f"    train: {len(os.listdir(f'{DATASET_PATH}/images/train'))}장")
print(f"    val:   {len(os.listdir(f'{DATASET_PATH}/images/val'))}장")


# ==========================================
# 4. YAML 생성
# ==========================================
with open(YAML_PATH, 'w') as f:
    f.write(f"""path: {DATASET_PATH}
train: images/train
val:   images/val

nc: 1
names: ['chair']
""")
print("[4] yaml 생성 완료!")


# ==========================================
# 5. 학습 (처음부터: yolov8s.pt 베이스)
# ==========================================
print("[5] 학습 시작...")

model = YOLO('yolov8s.pt')
results = model.train(
    data=YAML_PATH,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    project=f'{BASE_PATH}/chair_detect',
    name='exp',
)


# ==========================================
# 6. 모델 저장
# ==========================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_OUTPUT = f'{BASE_PATH}/chair_detect_{timestamp}.pt'

save_dir = str(results.save_dir)
best_pt  = os.path.join(save_dir, 'weights', 'best.pt')
print(f"[6] 학습 결과 경로: {save_dir}")

shutil.copy(best_pt, MODEL_OUTPUT)
print(f"[6] 모델 저장 완료: {MODEL_OUTPUT}")


# ==========================================
# 7. (선택) Google Drive에 저장
#    저장 위치 결정 후 주석 해제
# ==========================================
# from google.colab import drive
# drive.mount('/content/drive')
# os.makedirs('/content/drive/MyDrive/BORI', exist_ok=True)
# shutil.copy(MODEL_OUTPUT, '/content/drive/MyDrive/BORI/best.pt')
# print("Drive 저장 완료!")
