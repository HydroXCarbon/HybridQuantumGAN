from setuptools import setup, find_packages

setup(
    name='QuantumGAN',
    version='0.1.0',
    author="Purin Pongpanich",
    author_email="purin.pongpanich@gmail.com",
    url="https://github.com/HydroXCarbon/QuantumGAN",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        'matplotlib==3.9.0',
        "torchvision==0.18.1",
        "torch==2.3.1",
        "qiskit==1.1.1",
        "qiskit-aer==0.14.2",
        "qiskit-ibm-runtime==0.24.1",
        "python-dotenv==1.0.1",
        "pylatexenc==2.10",
        "numpy==1.26.4",
        "ibm-cloud-sdk-core==3.20.1",
        "ibm-platform-services==0.54.1",
        "nvidia-cublas-cu12==12.1.3.1",
        "nvidia-cuda-cupti-cu12==12.1.105",
        "nvidia-cuda-nvrtc-cu12==12.1.105",
        "nvidia-cuda-runtime-cu12==12.1.105",
        "nvidia-cudnn-cu12==8.9.2.26",
        "nvidia-cufft-cu12==11.0.2.54",
        "nvidia-curand-cu12==10.3.2.106",
        "nvidia-cusolver-cu12==11.4.5.107",
        "nvidia-cusparse-cu12==12.1.0.106",
        "nvidia-nccl-cu12==2.20.5",
        "nvidia-nvjitlink-cu12==12.5.82",
        "nvidia-nvtx-cu12==12.1.105",
        "packaging==24.1",
        "qiskit-algorithms==0.3.0",
        "qiskit_machine_learning==0.7.2"
    ]
)