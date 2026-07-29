from setuptools import setup, find_packages

setup(
    name="firefly-node",
    version="0.6.0",
    description="萤火虫火种客户端 · 联邦微调训练工具",
    author="Firefly LM",
    author_email="admin@firefly-lm.com",
    url="https://github.com/firefly-lm-org/firefly-client",
    packages=find_packages(where="app") if False else find_packages(),
    package_dir={"": "."},
    install_requires=[
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate",
        "datasets",
        "sentencepiece",
        "typer[all]",
        "httpx",
        "rich",
        "bcrypt",
        "pyjwt",
        "cryptography",
    ],
    extras_require={
        "gpu": [
            "torch",
            "unsloth @ git+https://github.com/unslothai/unsloth.git",
        ],
    },
    entry_points={
        "console_scripts": [
            "firefly-node=app.main:app",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
