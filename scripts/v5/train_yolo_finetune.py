import argparse
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning YOLO sur MoCA (classe unique animal).")
    parser.add_argument("--data-yaml", type=str, default="data/MoCA_YOLO/moca_yolo.yaml")
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="outputs/yolo/models/yolo_zeroshot_pretrained_coco/yolov8n.pt",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None, help="cpu, mps, 0, 0,1 ...")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--run-name", type=str, default="yolo_finetune_moca_run1")
    parser.add_argument("--project-dir", type=str, default="outputs/yolo/models")
    return parser.parse_args()


def main():
    args = parse_args()

    data_yaml = Path(args.data_yaml)
    pretrained_model = Path(args.pretrained_model)
    project_dir = Path(args.project_dir)
    run_dir = project_dir / args.run_name

    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data yaml not found: {data_yaml}")
    if not pretrained_model.exists():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_model}")

    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(pretrained_model))
    train_results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        project=str(project_dir),
        name=args.run_name,
        exist_ok=True,
        verbose=True,
        pretrained=True,
        plots=True,
    )

    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_yaml": str(data_yaml),
        "pretrained_model": str(pretrained_model),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "project_dir": str(project_dir),
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "best_pt_exists": best_pt.exists(),
        "best_pt": str(best_pt),
        "last_pt_exists": last_pt.exists(),
        "last_pt": str(last_pt),
        "results_object_type": str(type(train_results)),
    }

    summary_path = run_dir / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Training completed. Run dir: {run_dir}")
    print(f"Best checkpoint: {best_pt}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
