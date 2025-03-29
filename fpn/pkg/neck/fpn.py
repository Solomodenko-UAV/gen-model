import os 

is_cpu = os.environ.get('CPU_ONLY', '0') == '1'
if is_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage
    