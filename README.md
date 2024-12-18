# py3.11
```bash
sudo apt install cuda-toolkit-11-8
sudo apt install libcudnn8=8.6.0.163+cuda11.8
conda create -n clm python=3.10
conda activate clm
pip download torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu --no-deps -d wheels
pip install . -r requirements.txt
# test gpu
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
