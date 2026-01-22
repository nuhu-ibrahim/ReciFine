import os
import re
from glob import glob

from setuptools import find_packages, setup

data_files = []
for root, dirs, files in os.walk("configs"):
    if files:
        data_files.append(
            (os.path.relpath(root, "configs"), [os.path.join(root, f) for f in files])
        )

scripts = []
for fname in glob("bin/*"):
    with open(fname, "r", encoding="utf-8") as fh:
        first = fh.readline()
        if re.search(r"^#!.*python", first):
            scripts.append(fname)

def read_requirements(path: str) -> list[str]:
    reqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs


setup(
    name="ReciFine",
    version="1.0",
    description="The ReciFine Library provides code, training scripts, inference wrappers, and documentation for recipe-focused Named Entity Recognition (NER) using traditional BIO-n tagging and knowledge-augmented & entity type-specific token classification.",
    author="Nuhu Ibrahim",
    author_email="nuhu.ibrahim@manchester.ac.uk",
    packages=find_packages(where="src", exclude=("tests", "test",)),
    package_dir={"": "src"},
    install_requires=read_requirements("requirements.txt"),
    include_package_data=True,          
    data_files=data_files,              
    scripts=scripts,                    
    # python_requires=">=3.10,<3.13",     
)
