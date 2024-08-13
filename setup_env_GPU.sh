#! /bin/env bash

export VENV_DIR=".venv"

if [ ! -d ${VENV_DIR} ]; then
    python -m venv ${VENV_DIR}
fi

source ${VENV_DIR}/bin/activate

pip install uv
uv pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt