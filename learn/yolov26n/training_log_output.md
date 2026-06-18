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
    checkpoint_path = 'yolo26n.pt'

# Load a pretrained YOLO model
model = YOLO(checkpoint_path)
# Train the model
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=25,
    imgsz=640,
    batch=64,
    # lr0=0.0001,
    # freeze=10,
    save=True,
    project=drive_save_path, # Redirects the output to your Drive
    name=directory,
    amp=True,
    cache=True,
    patience=3,
    optimizer='AdamW',      # Often performs better for detection
    rect=True,              # Rectangular training reduces padding inefficiency
    workers=8,              # Number of CPU threads for data loading
    exist_ok=True
)
```



Output:

```
loading Roboflow workspace...
loading Roboflow project...
Checkpoint found at yolo26n.pt. Resuming training...
Ultralytics 8.4.70 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (NVIDIA A100-SXM4-40GB, 40441MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=64, bgr=0.0, box=7.5, cache=True, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/content/fire-and-smoke-detection-2/data.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=25, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo26n.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=trial_yolo_fire_smoke, nbs=64, nms=False, opset=None, optimize=False, optimizer=AdamW, overlap_mask=True, patience=3, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/content/drive/MyDrive/YOLOn_Checkpoints/trial_yolo_fire_smoke, rect=True, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/drive/MyDrive/YOLOn_Checkpoints/trial_yolo_fire_smoke/trial_yolo_fire_smoke, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=2

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5, 3, True]        
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    119808  ultralytics.nn.modules.block.C3k2            [384, 128, 1, True]           
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1     34304  ultralytics.nn.modules.block.C3k2            [256, 64, 1, True]            
 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1     95232  ultralytics.nn.modules.block.C3k2            [192, 128, 1, True]           
 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1    463104  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True, 0.5, True]
 23        [16, 19, 22]  1    241956  ultralytics.nn.modules.head.Detect           [2, 1, True, [64, 128, 256]]  
YOLO26n summary: 260 layers, 2,504,580 parameters, 2,504,580 gradients, 5.8 GFLOPs

Transferred 606/708 items from pretrained weights
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3408.1±1588.1 MB/s, size: 212.4 KB)
train: Scanning /content/fire-and-smoke-detection-2/train/labels.cache... 13423 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 13423/13423 4.0Git/s 0.0s
WARNING ⚠️ cache='ram' may produce non-deterministic training results. Consider cache='disk' as a deterministic alternative if your disk space allows.
train: Caching images (8.6GB RAM): 100% ━━━━━━━━━━━━ 13423/13423 712.9it/s 18.8s
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1717.0±1271.4 MB/s, size: 229.6 KB)
val: Scanning /content/fire-and-smoke-detection-2/valid/labels.cache... 1277 images, 1 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1277/1277 111.6Mit/s 0.0s
WARNING ⚠️ cache='ram' may produce non-deterministic training results. Consider cache='disk' as a deterministic alternative if your disk space allows.
val: Caching images (0.8GB RAM): 100% ━━━━━━━━━━━━ 1277/1277 209.5it/s 6.1s
optimizer: AdamW(lr=0.01, momentum=0.937) with parameter groups 114 weight(decay=0.0), 126 weight(decay=0.0005), 126 bias(decay=0.0)
Plotting labels to /content/drive/MyDrive/YOLOn_Checkpoints/trial_yolo_fire_smoke/trial_yolo_fire_smoke/labels.jpg... 
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to /content/drive/MyDrive/YOLOn_Checkpoints/trial_yolo_fire_smoke/trial_yolo_fire_smoke
Starting training for 25 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/25      11.1G      1.505      2.563    0.01312        144        640: 100% ━━━━━━━━━━━━ 210/210 3.1it/s 1:09
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 1.1it/s 9.3s
                   all       1277       3931       0.14     0.0971     0.0545     0.0128

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/25      11.1G       1.51      1.894    0.01328        144        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.6it/s 2.8s
                   all       1277       3931     0.0817       0.13     0.0503     0.0127

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/25      11.1G      1.494      1.778    0.01306        143        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.6it/s 2.8s
                   all       1277       3931      0.132      0.159     0.0737     0.0187

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/25      11.1G       1.44      1.659    0.01243        152        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.4it/s 2.9s
                   all       1277       3931      0.222      0.188      0.123     0.0351

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/25      11.1G      1.386      1.531    0.01188        128        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.5it/s 2.9s
                   all       1277       3931      0.271      0.258       0.17     0.0477

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/25      11.1G       1.34       1.47     0.0114        130        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.4it/s 2.9s
                   all       1277       3931      0.256      0.233      0.165     0.0484

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/25      11.1G      1.303      1.392    0.01103        129        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.4it/s 3.0s
                   all       1277       3931      0.326      0.209      0.172     0.0562

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/25      11.1G      1.283      1.331    0.01081        147        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.7it/s 2.7s
                   all       1277       3931      0.157     0.0787     0.0573     0.0161

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/25      11.1G      1.255      1.297    0.01046        136        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.3it/s 3.0s
                   all       1277       3931      0.299      0.256      0.194      0.063

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/25      11.1G      1.236      1.272    0.01033        148        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.5it/s 2.9s
                   all       1277       3931       0.41      0.202      0.199     0.0643

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/25      11.1G      1.215      1.237    0.01009        136        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.4it/s 2.9s
                   all       1277       3931      0.352      0.304      0.246     0.0811

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/25      11.1G      1.198      1.183   0.009933        140        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 53.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.3it/s 3.0s
                   all       1277       3931      0.335        0.3      0.242     0.0797

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/25      11.1G      1.179      1.157   0.009723        130        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.4it/s 3.0s
                   all       1277       3931      0.377      0.338      0.297      0.103

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/25      11.1G      1.171      1.139   0.009679        155        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.3it/s 3.0s
                   all       1277       3931      0.436      0.354      0.332      0.116

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/25      11.1G      1.153      1.113   0.009416        125        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.4it/s 2.9s
                   all       1277       3931      0.386      0.333      0.299      0.103
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/25      11.1G      1.144      1.097   0.009396        146        640: 100% ━━━━━━━━━━━━ 210/210 3.8it/s 55.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.4it/s 2.9s
                   all       1277       3931      0.425      0.405      0.359      0.121

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/25      11.1G      1.129      1.066   0.009159        153        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.4it/s 2.9s
                   all       1277       3931      0.418      0.373      0.335      0.116

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/25      11.1G      1.106      1.033    0.00894        161        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.3it/s 3.0s
                   all       1277       3931      0.463      0.401      0.385      0.139

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/25      11.1G      1.089      1.022   0.008818        153        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.2it/s 3.1s
                   all       1277       3931      0.498      0.392      0.398      0.145

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/25      11.1G      1.085     0.9968   0.008796        129        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.4it/s 2.9s
                   all       1277       3931      0.477       0.43      0.416      0.154

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/25      11.1G      1.067     0.9743   0.008628        143        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 53.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.3it/s 3.0s
                   all       1277       3931      0.506      0.422      0.423      0.155

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/25      11.1G       1.06     0.9507   0.008575        146        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.3it/s 3.0s
                   all       1277       3931       0.51      0.434      0.434      0.159

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/25      11.1G       1.05     0.9199   0.008476        139        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.3it/s 3.0s
                   all       1277       3931      0.515      0.447      0.441      0.164

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/25      11.1G      1.048     0.9218   0.008369        167        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.3it/s 3.0s
                   all       1277       3931       0.53      0.443      0.452       0.17

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/25      11.1G      1.031     0.8924   0.008219        142        640: 100% ━━━━━━━━━━━━ 210/210 4.0it/s 52.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 3.3it/s 3.0s
                   all       1277       3931      0.532      0.454      0.454      0.169

25 epochs completed in 0.399 hours.
Optimizer stripped from /content/drive/MyDrive/YOLOn_Checkpoints/trial_yolo_fire_smoke/trial_yolo_fire_smoke/weights/last.pt, 5.4MB
Optimizer stripped from /content/drive/MyDrive/YOLOn_Checkpoints/trial_yolo_fire_smoke/trial_yolo_fire_smoke/weights/best.pt, 5.4MB

Validating /content/drive/MyDrive/YOLOn_Checkpoints/trial_yolo_fire_smoke/trial_yolo_fire_smoke/weights/best.pt...
Ultralytics 8.4.70 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (NVIDIA A100-SXM4-40GB, 40441MiB)
YOLO26n summary (fused): 122 layers, 2,375,226 parameters, 0 gradients, 5.2 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 2.1it/s 4.7s
                   all       1277       3931      0.529      0.443      0.452       0.17
                  fire        944       2121      0.528      0.548      0.531      0.198
                 smoke        980       1810      0.529      0.338      0.372      0.141
Speed: 0.1ms preprocess, 0.3ms inference, 0.0ms loss, 0.1ms postprocess per image
Results saved to /content/drive/MyDrive/YOLOn_Checkpoints/trial_yolo_fire_smoke/trial_yolo_fire_smoke

```

