Connect to Lambda
```bash
ssh -i '<SSH-KEY-FILE-PATH>' ubuntu@<INSTANCE-IP>
```

Follow the guide: https://docs.lambda.ai/public-cloud/on-demand/managing-system-environment/
1. uname -m
2. If aarch: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
3. If x86: curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
4. sh Miniconda3-latest-Linux-*.sh
5. source ~/.bashrc
6. conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
7. conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

Install packages
```bash
conda create -n guardian-loop python=3.10 -y
conda activate guardian-loop
conda install -c conda-forge numpy pandas scikit-learn scipy matplotlib ipykernel yaml -y
pip3 install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install openai langchain langgraph transformers sentence-transformers datasets
```

Copy outputs
```bash
scp -i ~/keys/mykey.pem ubuntu@54.123.45.67:/home/ubuntu/output.json ~/Downloads/
```