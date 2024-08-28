import setuptools
 
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
 
setuptools.setup(
    name="tf-vit",
    version="1.0.1",
    author="AryaKoureshi",
    author_email="arya.koureshi@gmail.com",
    description="Implementation of ViT model based on TF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AryaKoureshi/TF-ViT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
