import datetime
import time
import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import utils
from coco_utils import get_dataset
from engine import evaluate
from dg_fasterrcnn import DGFasterRCNN

def get_transform(args):
    return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets/DFC2023/track1/", type=str, help="dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument('--model-path', type=str, default='best_map@50.pth', help='saved model path')
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    parser.add_argument('--img-dg', action="store_true", help="whether the image level domain generalization is included during training")
    parser.add_argument('--ins-dg', action="store_true", help="whether the box level domain generalization is included during training")

    return parser


def main(args):
    if args.backend.lower() == "datapoint" and not args.use_v2:
        raise ValueError("Use --use-v2 if you want to use the datapoint backend.")

    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset_test, num_classes, num_domains = get_dataset(args.data_path, transforms=get_transform(args), ann_folder="annotations/1classes", image_set='test')


    print("Creating data loader")

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    print("Loading model")

    model = DGFasterRCNN(num_classes=num_classes+1, num_domains=num_domains, img_dg=args.img_dg, ins_dg=args.ins_dg)

    model.to(device)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])


    print("Start testing")
    start_time = time.time()

    coco_evaluator = evaluate(model, data_loader_test, device=device)

    coco_evaluator.coco_eval['bbox'].stats[1]


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Inference time {total_time_str}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)