```python
from roboflow import Roboflow
from ultralytics import YOLO

# Initialize Roboflow and download the dataset
rf = Roboflow(api_key=userdata.get('roboflow_api'))
project = rf.workspace("middle-east-tech-university").project("fire-and-smoke-detection-hiwia")
version = project.version(2) # Change version number if needed
dataset = version.download("yolo26")

# Logic to load model
if os.path.exists(checkpoint_path):
    print(f"Checkpoint found at {checkpoint_path}. Resuming training...")
    
else:
    checkpoint_path = 'yolo26x.pt'

# Load a pretrained YOLO model
model = YOLO(checkpoint_path)
# Train the model
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=25,
    imgsz=640,
    batch=32,
    # lr0=0.0001,
    # freeze=10,
    save=True,
    project=drive_save_path, # Redirects the output to your Drive
    name=directory,
    amp=True,
    cache=True,
    patience=10,
    optimizer='AdamW',      # Often performs better for detection
    rect=True,              # Rectangular training reduces padding inefficiency
    workers=8,              # Number of CPU threads for data loading
    exist_ok=True
)
```



Output:

```
Creating new Ultralytics Settings v0.0.6 file ✅ 
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
loading Roboflow workspace...
loading Roboflow project...
Downloading Dataset Version Zip in fire-and-smoke-detection-2 to yolo26:: 100%|██████████| 3241404/3241404 [00:44<00:00, 73096.28it/s]

Extracting Dataset Version Zip to fire-and-smoke-detection-2 in yolo26:: 100%|██████████| 30686/30686 [00:08<00:00, 3516.02it/s]
Checkpoint found at /content/drive/MyDrive/YOLOx_Checkpoints/trial_yolo_fire_smoke-v2/trial_yolo_fire_smoke-v2/weights/last.pt. Resuming training...
Ultralytics 8.4.70 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (NVIDIA A100-SXM4-40GB, 40441MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=32, bgr=0.0, box=7.5, cache=True, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/content/fire-and-smoke-detection-2/data.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=25, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=/content/drive/MyDrive/YOLOx_Checkpoints/trial_yolo_fire_smoke-v2/trial_yolo_fire_smoke-v2/weights/last.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=trial_yolo_fire_smoke-v2, nbs=64, nms=False, opset=None, optimize=False, optimizer=AdamW, overlap_mask=True, patience=10, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/content/drive/MyDrive/YOLOx_Checkpoints/trial_yolo_fire_smoke-v2, rect=True, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/drive/MyDrive/YOLOx_Checkpoints/trial_yolo_fire_smoke-v2/trial_yolo_fire_smoke-v2, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 15.6MB/s 0.0s

                   from  n    params  module                                       arguments                     
  0                  -1  1      2784  ultralytics.nn.modules.conv.Conv             [3, 96, 3, 2]                 
  1                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  2                  -1  2    389760  ultralytics.nn.modules.block.C3k2            [192, 384, 2, True, 0.25]     
  3                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
  4                  -1  2   1553664  ultralytics.nn.modules.block.C3k2            [384, 768, 2, True, 0.25]     
  5                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  6                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  7                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  8                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  9                  -1  1   1476864  ultralytics.nn.modules.block.SPPF            [768, 768, 5, 3, True]        
 10                  -1  2   3264768  ultralytics.nn.modules.block.C2PSA           [768, 768, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2   1700352  ultralytics.nn.modules.block.C3k2            [1536, 384, 2, True]          
 17                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   5317632  ultralytics.nn.modules.block.C3k2            [1152, 768, 2, True]          
 20                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   4436736  ultralytics.nn.modules.block.C3k2            [1536, 768, 1, True, 0.5, True]
 23        [16, 19, 22]  1   6260772  ultralytics.nn.modules.head.Detect           [2, 1, True, [384, 768, 768]] 
YOLO26x summary: 392 layers, 58,813,188 parameters, 58,813,188 gradients, 208.5 GFLOPs

Transferred 1092/1092 items from pretrained weights
AMP: running Automatic Mixed Precision (AMP) checks...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 73.0MB/s 0.1s
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2442.0±1853.9 MB/s, size: 210.8 KB)
train: Scanning /content/fire-and-smoke-detection-2/train/labels... 13423 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 13423/13423 1.5Kit/s 8.7s
train: New cache created: /content/fire-and-smoke-detection-2/train/labels.cache
WARNING ⚠️ cache='ram' may produce non-deterministic training results. Consider cache='disk' as a deterministic alternative if your disk space allows.
train: Caching images (8.6GB RAM): 100% ━━━━━━━━━━━━ 13423/13423 717.5it/s 18.7s
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1374.2±995.5 MB/s, size: 222.6 KB)
val: Scanning /content/fire-and-smoke-detection-2/valid/labels... 1277 images, 1 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1277/1277 1.1Kit/s 1.1s
val: New cache created: /content/fire-and-smoke-detection-2/valid/labels.cache
WARNING ⚠️ cache='ram' may produce non-deterministic training results. Consider cache='disk' as a deterministic alternative if your disk space allows.
val: Caching images (0.8GB RAM): 100% ━━━━━━━━━━━━ 1277/1277 538.9it/s 2.4s
optimizer: AdamW(lr=0.01, momentum=0.937) with parameter groups 178 weight(decay=0.0), 190 weight(decay=0.0005), 190 bias(decay=0.0)
Plotting labels to /content/drive/MyDrive/YOLOx_Checkpoints/trial_yolo_fire_smoke-v2/trial_yolo_fire_smoke-v2/labels.jpg... 
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to /content/drive/MyDrive/YOLOx_Checkpoints/trial_yolo_fire_smoke-v2/trial_yolo_fire_smoke-v2
Starting training for 25 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/25      31.6G      1.324      1.402    0.01157         53        640: 100% ━━━━━━━━━━━━ 420/420 1.8it/s 3:50
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 1.8it/s 11.2s
                   all       1277       3931      0.172      0.157      0.112     0.0329

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/25      32.1G      1.337      1.436    0.01178         45        640: 100% ━━━━━━━━━━━━ 420/420 2.0it/s 3:29
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.7s
                   all       1277       3931        0.4      0.197      0.198     0.0629

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/25      32.1G      1.314        1.4     0.0115         46        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:24
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.6it/s 5.6s
                   all       1277       3931     0.0842      0.113     0.0309    0.00866

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/25      32.1G      1.313      1.395    0.01147         49        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:24
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.4it/s 5.8s
                   all       1277       3931      0.148       0.27     0.0824     0.0255

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/25      32.1G      1.272      1.339    0.01107         43        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.8s
                   all       1277       3931      0.327      0.269      0.213     0.0713

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/25      32.1G      1.236      1.287    0.01069         53        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:24
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.4it/s 5.8s
                   all       1277       3931      0.146      0.284     0.0847     0.0259

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/25      32.1G      1.206      1.247    0.01037         49        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:24
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.7s
                   all       1277       3931      0.356      0.234       0.19     0.0639

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/25      32.1G      1.186      1.229     0.0101         39        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.8s
                   all       1277       3931      0.324      0.268      0.216     0.0731

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/25      32.1G      1.177      1.203    0.01005         54        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.7s
                   all       1277       3931      0.397      0.298      0.261     0.0865

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/25      32.1G      1.158      1.159   0.009919         43        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.7s
                   all       1277       3931      0.418      0.316      0.296     0.0987

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/25      32.1G      1.145      1.142   0.009801         35        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.8s
                   all       1277       3931      0.426       0.33      0.312      0.108

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/25      32.1G      1.121      1.103   0.009501         44        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.8s
                   all       1277       3931      0.401       0.32      0.293     0.0986

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/25      32.1G      1.105       1.08   0.009452         42        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.7s
                   all       1277       3931      0.446      0.352      0.342      0.121

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/25      32.1G      1.099      1.073   0.009328         46        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.8s
                   all       1277       3931      0.448      0.364       0.35      0.123

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/25      32.1G      1.078       1.05    0.00906         36        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.7s
                   all       1277       3931      0.447       0.33      0.331      0.118
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/25      32.1G      1.063      1.038   0.009005         42        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:24
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.4it/s 5.8s
                   all       1277       3931      0.454       0.35      0.345      0.124

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/25      32.1G      1.053      1.019   0.008821         37        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.7s
                   all       1277       3931      0.451      0.363      0.354      0.128

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/25      32.1G      1.044     0.9999   0.008677         52        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.8s
                   all       1277       3931      0.445      0.367      0.351      0.128

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/25      32.1G      1.024     0.9791   0.008589         46        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.4it/s 5.8s
                   all       1277       3931      0.454      0.367       0.36      0.131

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/25      32.1G      1.011      0.948   0.008489         50        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.4it/s 5.8s
                   all       1277       3931      0.472      0.388      0.376      0.138

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/25      32.1G      1.001      0.942   0.008375         55        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:24
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.4it/s 5.9s
                   all       1277       3931      0.475      0.374      0.372      0.137

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/25      32.1G     0.9931     0.9158   0.008288         40        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.4it/s 5.8s
                   all       1277       3931      0.477      0.388      0.386      0.143

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/25      32.1G     0.9809     0.8972   0.008154         44        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:24
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.7s
                   all       1277       3931      0.486      0.397      0.396      0.148

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/25      32.1G     0.9791     0.9077    0.00806         60        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:24
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.4it/s 5.9s
                   all       1277       3931      0.485      0.385      0.386      0.143

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/25      32.1G     0.9598     0.8884    0.00796         41        640: 100% ━━━━━━━━━━━━ 420/420 2.1it/s 3:23
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 3.5it/s 5.7s
                   all       1277       3931      0.497      0.402      0.404       0.15

25 epochs completed in 1.485 hours.
Optimizer stripped from /content/drive/MyDrive/YOLOx_Checkpoints/trial_yolo_fire_smoke-v2/trial_yolo_fire_smoke-v2/weights/last.pt, 118.3MB
Optimizer stripped from /content/drive/MyDrive/YOLOx_Checkpoints/trial_yolo_fire_smoke-v2/trial_yolo_fire_smoke-v2/weights/best.pt, 118.3MB

Validating /content/drive/MyDrive/YOLOx_Checkpoints/trial_yolo_fire_smoke-v2/trial_yolo_fire_smoke-v2/weights/best.pt...
Ultralytics 8.4.70 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (NVIDIA A100-SXM4-40GB, 40441MiB)
YOLO26x summary (fused): 190 layers, 55,635,858 parameters, 0 gradients, 193.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 20/20 2.9it/s 6.8s
                   all       1277       3931      0.495      0.402      0.403       0.15
                  fire        944       2121      0.526      0.493      0.488      0.188
                 smoke        980       1810      0.463      0.311      0.319      0.112
Speed: 0.1ms preprocess, 2.4ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /content/drive/MyDrive/YOLOx_Checkpoints/trial_yolo_fire_smoke-v2/trial_yolo_fire_smoke-v2

```

