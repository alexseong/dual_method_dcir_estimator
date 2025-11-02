from setuptools import setup, find_packages

setup(
    name="dual_method_dcir_estimator",
    version="0.1.0",
    description="Temperature-aware 2RC Neural-ODE (RK4 + Residual NN) for DCIR",
    author="Alex Seong, Shobhan Pujari",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.2",
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "scikit-learn>=1.3",
        "einops>=0.7",
        "pyarrow>=15.0",
    ],
)