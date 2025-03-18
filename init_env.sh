conda create -n emigm python==3.10
conda activate emigm

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

mkdir others 
cd others

git clone https://github.com/mit-han-lab/efficientvit.git
cd efficientvit
pip install -r requirements.txt --no-deps
pip install -e . --no-deps 
cd ..
pip uninstall torch-fidelity -y

pip install rich
pip install --upgrade omegaconf
pip install --upgrade huggingface_hub
pip install timm==0.9.12 --no-deps
pip install gdown==5.2.0 --no-deps
pip install opencv-python --no-deps
pip install torchdiffeq
pip install tensorboard
pip install atorch

git clone https://github.com/LTH14/torch-fidelity.git
cd torch-fidelity
pip install -e .
cd ..

cd ..

