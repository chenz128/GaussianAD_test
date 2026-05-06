export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/opt/miniconda/envs/GaussianAD/lib:/usr/local/cuda-11.8/lib64:/opt/miniconda/envs/GaussianAD/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=8