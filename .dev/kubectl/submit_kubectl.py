import argparse
import os
import os.path as osp
import re
import tempfile
from pathlib import Path


def scandir(dir_path, suffix=None, recursive=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def parse_args():
    parser = argparse.ArgumentParser(description="Submit to nautilus via kubectl")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("job", help="kubectl config path")
    parser.add_argument("--branch", "-b", type=str, default="dev", help="git clone branch")
    parser.add_argument("--ln-exp", "-l", action="store_true", help="link experiment directory")
    parser.add_argument("--gpus", type=int, default=4, help="number of gpus to use ")
    parser.add_argument("--cpus", type=int, default=8, help="number of cpus to use")
    parser.add_argument("--mem", type=int, default=30, help="amount of memory to use")
    parser.add_argument(
        '--disk', type=int, default=150, help='amount of disk to use')
    parser.add_argument("--file", "-f", type=str, help="config txt file")
    parser.add_argument(
        "--name-space",
        "-n",
        type=str,
        default="self-supervised-video",
        choices=["self-supervised-video", "ece3d-vision", "image-model"],
    )
    parser.add_argument("--copy", action='store_true', help='whether copy imagenet data')
    args, rest = parser.parse_known_args()

    return args, rest


def submit(config, args, rest):
    py_args = " ".join(rest)
    script = "tools/dist_launch.sh"
    copy_script = ''
    if args.copy:
        copy_script += 'mkdir -p /mnt/dest/imagenet; gsutil -m rsync -erCUP /mnt/source/imagenet /mnt/dest/imagenet;' * 2  # noqa
    template_dict = dict(
        job_name=osp.splitext(osp.basename(config))[0].lower().replace("_", "-") + "-",
        name_space=args.name_space,
        branch=args.branch,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=f"{args.mem}Gi",
        max_cpus=int(args.cpus * 1.2),
        max_mem=f"{int(args.mem * 1.2)}Gi",
        config=config,
        script=script,
        py_args=py_args,
        copy_script=copy_script,
        ephemeral_storage=args.disk if args.copy else 10,
        link="ln -s /exps/deit/work_dirs; " if args.ln_exp else "",
        data_path='dst' if len(copy_script) else 'src')
    with open(args.job, "r") as f:
        config_file = f.read()
    for key, value in template_dict.items():
        regexp = r"\{\{\s*" + str(key) + r"\s*\}\}"
        config_file = re.sub(regexp, str(value), config_file)
    temp_config_file = tempfile.NamedTemporaryFile(suffix=osp.splitext(args.job)[1])
    with open(temp_config_file.name, "w") as tmp_config_file:
        tmp_config_file.write(config_file)
    os.system(f"kubectl create -f {temp_config_file.name}")
    tmp_config_file.close()


def main():
    args, rest = parse_args()
    if osp.isdir(args.config):
        if args.file is not None:
            with open(args.file) as f:
                submit_cfg_names = [line.strip() for line in f.readlines()]
            for cfg in scandir(args.config, recursive=True):
                if osp.basename(cfg) in submit_cfg_names:
                    submit(osp.join(args.config, cfg), args, rest)
        else:
            for cfg in scandir(args.config, suffix=".py"):
                if "playground" in cfg:
                    continue
                submit(osp.join(args.config, cfg), args, rest)
    else:
        submit(args.config, args, rest)


if __name__ == "__main__":
    main()
