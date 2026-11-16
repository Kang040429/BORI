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
#   python chairDetect_train.py --video chair.mp4 --pretrained best.pt --frame_interval 15
# ==========================================
parser = argparse.ArgumentParser(description='BORI 의자 YOLO 학습')
parser.add_argument('--epochs',         type=int,   default=10,   help='학습 epoch 수 (기본값: 10)')
parser.add_argument('--batch',          type=int,   default=16,   help='배치 크기 (기본값: 16)')
parser.add_argument('--val_count',      type=int,   default=1000, help='val 이미지 수 (기본값: 1000)')
parser.add_argument('--device',         type=str,   default='0',  help='디바이스: 0=GPU, cpu=CPU (기본값: 0)')
parser.add_argument('--video',          type=str,   default=None, help='추가 학습용 영상 경로 (없으면 스킵)')
parser.add_argument('--pretrained',     type=str,   default=None, help='영상 자동 라벨링에 쓸 기존 모델 경로')
parser.add_argument('--frame_interval', type=int,   default=15,   help='영상에서 몇 프레임마다 추출할지 (기본값: 15)')
parser.add_argument('--conf',           type=float, default=0.3,  help='자동 라벨링 confidence 임계값 (기본값: 0.3)')
args, _ = parser.parse_known_args()  # Colab/Kaggle 호환용

EPOCHS         = args.epochs
BATCH_SIZE     = args.batch
VAL_COUNT      = args.val_count
DEVICE         = args.device if args.device == 'cpu' else int(args.device)
VIDEO_PATH     = args.video
PRETRAINED     = args.pretrained
FRAME_INTERVAL = args.frame_interval
CONF_THRESH    = args.conf

print("=" * 40)
print(f"  epochs         : {EPOCHS}")
print(f"  batch          : {BATCH_SIZE}")
print(f"  val_count      : {VAL_COUNT}")
print(f"  device         : {DEVICE}")
print(f"  video          : {VIDEO_PATH or '없음 (스킵)'}")
print(f"  pretrained     : {PRETRAINED or '없음'}")
print(f"  frame_interval : {FRAME_INTERVAL}")
print(f"  conf           : {CONF_THRESH}")
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
    BASE_PATH       = '/kaggle/working'
    COCO_IMAGE_PATH = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017'
    ANNOTATION_PATH = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/instances_train2017.json'
    USE_LOCAL_IMAGES = True

elif ENV == 'colab':
    BASE_PATH       = '/content'
    COCO_IMAGE_PATH = None
    ANNOTATION_PATH = f'{BASE_PATH}/annotations/instances_train2017.json'
    USE_LOCAL_IMAGES = False

else:  # 젯슨, 로컬
    BASE_PATH       = os.path.dirname(os.path.abspath(__file__))
    COCO_IMAGE_PATH = None
    ANNOTATION_PATH = f'{BASE_PATH}/annotations/instances_train2017.json'
    USE_LOCAL_IMAGES = False

DATASET_PATH = f'{BASE_PATH}/dataset'
YAML_PATH    = f'{BASE_PATH}/chair_coco.yaml'
IMG_SIZE     = 640

# 영상 경로 자동 설정 (--video 없을 때)
if VIDEO_PATH is None:
    if ENV == 'kaggle':
        # Kaggle Dataset으로 올린 경우 자동 탐색
        for root, dirs, files in os.walk('/kaggle/input'):
            for f in files:
                if f.endswith('.mp4'):
                    VIDEO_PATH = os.path.join(root, f)
                    print(f"[영상] 자동 감지: {VIDEO_PATH}")
                    break
    elif ENV == 'colab':
        # Google Drive 마운트 후 경로
        drive_path = '/content/drive/MyDrive/20261117_002118.mp4'
        if os.path.exists(drive_path):
            VIDEO_PATH = drive_path
            print(f"[영상] Drive에서 감지: {VIDEO_PATH}")

# 사전 학습 모델 경로 자동 설정 (--pretrained 없을 때)
if PRETRAINED is None:
    if ENV == 'kaggle':
        pt_path = '/kaggle/working/best.pt'
    elif ENV == 'colab':
        pt_path = '/content/drive/MyDrive/BORI/best.pt'
    else:
        pt_path = os.path.join(BASE_PATH, 'best.pt')
    if os.path.exists(pt_path):
        PRETRAINED = pt_path
        print(f"[모델] 사전 학습 모델 감지: {PRETRAINED}")


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
# 2.5. (선택) 영상 프레임 추출 + 자동 라벨링
#      --video 옵션 있을 때만 실행
# ==========================================
if VIDEO_PATH and os.path.exists(VIDEO_PATH):
    print(f"[2.5] 영상 프레임 추출 중: {VIDEO_PATH}")
    import cv2

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count, saved = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % FRAME_INTERVAL == 0:
            fname = f"custom_frame_{saved:05d}.jpg"
            cv2.imwrite(f"{DATASET_PATH}/images/train/{fname}", frame)
            saved += 1
        count += 1
    cap.release()
    print(f"    추출 완료: {saved}장")

    # 자동 라벨링 (기존 모델 있을 때)
    if PRETRAINED and os.path.exists(PRETRAINED):
        print(f"[2.5] 자동 라벨링 중 (모델: {PRETRAINED})...")
        label_model = YOLO(PRETRAINED)
        labeled = 0
        for i in range(saved):
            fname = f"custom_frame_{i:05d}.jpg"
            img_path = f"{DATASET_PATH}/images/train/{fname}"
            results = label_model.predict(img_path, conf=CONF_THRESH, verbose=False)
            boxes = results[0].boxes
            if len(boxes) == 0:
                # 의자 없으면 이미지/라벨 둘 다 제거
                os.remove(img_path)
                continue
            H, W = results[0].orig_shape
            lbl_path = f"{DATASET_PATH}/labels/train/{fname.replace('.jpg', '.txt')}"
            with open(lbl_path, 'w') as f:
                for box in boxes.xywhn:
                    cx, cy, bw, bh = box.tolist()
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            labeled += 1
        print(f"    라벨링 완료: {labeled}장 추가")
    else:
        print("    [주의] --pretrained 없음 → 프레임은 추가됐지만 라벨 없어서 학습에 미포함됨")
        print("    라벨링하려면 --pretrained best.pt 옵션 추가하세요")
elif VIDEO_PATH:
    print(f"[2.5] 영상 파일 없음: {VIDEO_PATH} → 스킵")
else:
    print("[2.5] 영상 없음, 스킵")


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
# 5. 학습
# ==========================================
print("[5] 학습 시작...")

model = YOLO('yolov8s.pt')
model.train(
    data=YAML_PATH,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    project=f'{BASE_PATH}/chair_detect',
    name='exp1',
)


# ==========================================
# 6. 모델 저장
# ==========================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_OUTPUT = f'{BASE_PATH}/chair_detect_{timestamp}.pt'

shutil.copy(f'{BASE_PATH}/chair_detect/exp1/weights/best.pt', MODEL_OUTPUT)
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
