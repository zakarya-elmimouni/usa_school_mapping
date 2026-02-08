from ultralytics import YOLO

# Path to your YAML file
#DATA_YAML = 'dataset/brazil/fixed_bb_and_manual_data/generated_dataset1/data.yaml'
DATA_YAML = 'dataset/usa/golden_data/data.yaml'
# Model: YOLO8n
model = YOLO('yolo12s.pt')

# Training
results = model.train(
    data=DATA_YAML,
    epochs=120,
    imgsz=500 ,
    batch=64,
    lr0=0.01,
    lrf=0.001,
    pretrained=True,
    seed=0,
    device=0,
    project='results/usa/rslt_yolo12s_golden_dataset',
    name='exp',
    save=True,
    plots=True,
    patience=8,  # early stopping
    save_period=20,
    verbose=True
    #translate=0.2,
    #degrees=15,
    #auto_augment=None
)

print(f"? Best model saved at: {results.save_dir}/weights/best.pt")
print(f"? All training plots saved at: {results.save_dir}")