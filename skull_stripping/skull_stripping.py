
import os
import shutil
import sys
from glob import glob
import nibabel as nib
import numpy as np
from pathlib import Path
import time 
import tempfile
import subprocess
from multiprocessing import Pool
import argparse
import torch 

def maybe_make_dir(path):
    os.makedirs(path, exist_ok=True)
    return Path(path)


def skull_stripping_by_mask(nifti_fn, mask_fn, out_file_path):
    """
    save nii.gz
    """
    # nifti_fn = input image
    # mask_fn = face mask
    # out_file_path = output image
    # Get the data arrays from the images
    nifti_image = nib.load(nifti_fn)
    mask_image = nib.load(mask_fn)
    nifti_data = nifti_image.get_fdata()
    mask_data = mask_image.get_fdata()

    # Apply the binary mask to the NIfTI data
    # inv_mask_data = 1 - mask_data
    masked_data = nifti_data * mask_data

    # Create a new NIfTI image with the masked data
    masked_image = nib.Nifti1Image(masked_data, nifti_image.affine)

    # Save the masked image to a new file
    nib.save(masked_image, f'{out_file_path}')


def infer_single(single_input_path, outputs_base_dir, cuda_id=0):
    """do inference on a single folder

    Args:
        input_path (path): input folder, where the 4 nii.gz are stored
        out_dir (path): out folder, where the seg.nii.gz is to be stored
        cuda_id (int): CUDA device ID to use
    """
    nnunetv1_rename_dict = {
        "-t1n.nii.gz": "_0001.nii.gz",
        "-t1c.nii.gz": "_0002.nii.gz",
        "-t2w.nii.gz": "_0003.nii.gz",
        "-t2f.nii.gz": "_0000.nii.gz",
        } 

    nnunetv2_rename_dict = {
        "-t1n.nii.gz": "_0000.nii.gz",
        "-t1c.nii.gz": "_0001.nii.gz",
        "-t2w.nii.gz": "_0002.nii.gz",
        "-t2f.nii.gz": "_0003.nii.gz",
        }


    maybe_make_dir(outputs_base_dir)

    with tempfile.TemporaryDirectory() as tmpdirname: # random tmp
        # tmpdirname = Path("/tmp/brats_skull_tmp")
        temp_dir = Path(tmpdirname)  # /tmp/tmp0xadd
        case_name = single_input_path.name           

        print(f'[CUDA:{cuda_id}] storing artifacts in tmp dir {temp_dir}')
        shutil.copytree(single_input_path, temp_dir, dirs_exist_ok=True)
        
        temp_dir_only_input_modalities = maybe_make_dir(temp_dir / 'input_mods')
        temp_dir_only_pred_mask = maybe_make_dir(temp_dir / 'pred_mask')
        
        for suffix, val in nnunetv1_rename_dict.items():
            os.rename(temp_dir/f"{case_name}{suffix}",  temp_dir_only_input_modalities/f"{case_name.strip('-000') }{val}")
        
        cmd = f'CUDA_VISIBLE_DEVICES={cuda_id} nnUNet_predict -i {temp_dir_only_input_modalities} -o {temp_dir_only_pred_mask} -t Task070_autosegm -m 3d_fullres'

        print(cmd)

        subprocess.run(cmd, shell=True) # nnUnet-v1 generates the mask file. 

        for suffix, val in nnunetv1_rename_dict.items():
            input_file_path = temp_dir_only_input_modalities / f"{case_name.strip('-000') }{val}"
            pred_mask_path = temp_dir_only_pred_mask / f"{case_name.strip('-000') }.nii.gz"

            # maybe_make_dir(outputs_base_dir / case_name)
            nnunet_v2_val = nnunetv2_rename_dict[suffix]
            output_file_path  = outputs_base_dir / f"{case_name}{nnunet_v2_val}" 
            ##  保存文件后缀为0000 00001

            skull_stripping_by_mask(input_file_path, pred_mask_path, output_file_path)
            print(f"[CUDA:{cuda_id}] skull stripped image saved to: {output_file_path}")


def process_batch(args_tuple):
    """处理一批文件的包装函数"""
    file_paths, outputs_base_dir, cuda_id = args_tuple
    
    for single_input_path in file_paths:
        case_start_time = time.time()
        print(f"[CUDA:{cuda_id}] Start deal with {single_input_path.name}")
        infer_single(single_input_path, outputs_base_dir, cuda_id)
        case_time = time.time() - case_start_time
        print(f"[CUDA:{cuda_id}] Case: {single_input_path.name} done. Time cost: {case_time:.2f} s")



def skull_stripping(inputs_base_dir, outputs_base_dir, cuda_ids=None):
    """
    Strip the skull from every case under inputs_base_dir and write the four
    modalities back out flat into outputs_base_dir, renamed to the nnU-Net v2
    channel order (_0000.._0003). Cases are split evenly across the GPUs.
    """

    if cuda_ids is None:
        if torch.cuda.is_available():
            cuda_ids = list(range(torch.cuda.device_count()))
        else:
            cuda_ids = [0]  # keep a single worker so the run still proceeds on CPU
            print("Warning: No CUDA devices available")

    inputs_base_dir = Path(inputs_base_dir)
    outputs_base_dir = Path(outputs_base_dir)
    # cuda_ids = args.cuda_ids
    
    # 获取所有输入文件夹
    all_input_paths = sorted(Path(inputs_base_dir).iterdir())
    total_files = len(all_input_paths)
    num_gpus = len(cuda_ids)
    
    print(f"Total files: {total_files}, Using {num_gpus} GPUs: {cuda_ids}")
    
    # 将文件平均分配给各个GPU
    files_per_gpu = total_files // num_gpus
    remainder = total_files % num_gpus
    
    batches = []
    start_idx = 0
    
    for i, cuda_id in enumerate(cuda_ids):
        # 前remainder个GPU多分配一个文件
        batch_size = files_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + batch_size
        
        batch_files = all_input_paths[start_idx:end_idx]
        batches.append((batch_files, outputs_base_dir, cuda_id))
        
        print(f"GPU {cuda_id}: {len(batch_files)} files (indices {start_idx}-{end_idx-1})")
        start_idx = end_idx
    
    # 并行处理
    start_time = time.time()
    
    with Pool(processes=num_gpus) as pool:
        pool.map(process_batch, batches)
    
    total_time = time.time() - start_time
    print(f"All cases completed. Total time: {total_time:.2f} s")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel skull stripping with nnU-Net v1 (Task070_autosegm) across GPUs."
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input base dir: one sub-folder per case, each holding the 4 '
                             'BraTS modality files (<case>-t1n/-t1c/-t2w/-t2f.nii.gz).')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output base dir. All cases are written flat as '
                             '<case>_0000.nii.gz .. <case>_0003.nii.gz (nnU-Net v2 order).')
    parser.add_argument('--cuda_ids', '-c', type=int, nargs='+', default=None,
                        help='CUDA device IDs to use. Default: every visible GPU.')
    args = parser.parse_args()

    skull_stripping(inputs_base_dir=args.input,
                    outputs_base_dir=args.output,
                    cuda_ids=args.cuda_ids)


if __name__ == "__main__":
    main()
