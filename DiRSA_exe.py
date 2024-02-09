import argparse
import torch
import datetime
import json
import yaml
import os

from DiRSA_model import DiRSA_IQ
from DiRSA_read import get_dataloader
from DiRSA_utils import train, evaluate

modtype=['8PSK','AM-DSB','AM_SSB','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','WBFM']
parser = argparse.ArgumentParser(description="DiRSA")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument('--device', default='cuda:1', help='Device for Attack')
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--testmissingratio", type=float, default=15/128)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)

args = parser.parse_args()
print(args)

path = args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")


iftrain = True
make_aug_dataset = False
if (make_aug_dataset):
    iftrain = False

args.modelfolder = "./save/pretrained/"
model_pre="0.05"
version_num="DiRSA"
model_name="DiRSAmodel"+model_pre+"-"+version_num
filepath_suff=model_pre+'-'+version_num
foldername = args.modelfolder+model_pre+"-"+version_num+"/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

if (make_aug_dataset):
    test_loaders = get_dataloader(
        seed=args.seed,
        nfold=args.nfold,
        batch_size=config["train"]["batch_size"],
        missing_ratio=config["model"]["test_missing_ratio"],
        make_aug_dataset=make_aug_dataset,
        sampling_scale=0.025,
        mix_k=4,
        new_data=False,
        test_scale=0.1,
        val_scale=0.1,
        filepath_suff=filepath_suff,
    )
else:
    train_loaders, valid_loaders = get_dataloader(
        seed=args.seed,
        nfold=args.nfold,
        batch_size=config["train"]["batch_size"],
        missing_ratio=config["model"]["test_missing_ratio"],
        make_aug_dataset=make_aug_dataset,
        sampling_scale=0.05,
        mix_k=2,
        new_data=True,
        test_scale=0.1,
        val_scale=0.1,
        filepath_suff=filepath_suff,
    )

model = DiRSA_IQ(config, args.device).to(args.device)

if iftrain:

    for i in range(0, len(modtype)):
        train(
            model,
            config["train"],
            train_loaders[i],

            valid_loader=valid_loaders[i],
            foldername=foldername,
            modelname=model_name+"-"+modtype[i]+".pth"
        )
else:
    evaluate(model, test_loaders, nsample=args.nsample, scaler=1, foldername=args.modelfolder, model_pre=model_pre, version_num=version_num, model_input=model_name)

