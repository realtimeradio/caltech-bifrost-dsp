import setuptools
import glob
import os

VERSION = "1.0.0"

desc_file = "README.md"
long_description = ""
if os.path.isfile(desc_file):
    with open(desc_file, "r") as fh:
        long_description = fh.read()

req_file = "requirements.txt"
install_requires = []
if os.path.isfile(req_file):
    with open(req_file, "r") as fh:
        install_requires = fh.read().splitlines()

setuptools.setup(
    name="lwa352-pipeline",
    version=VERSION,
    author="Real-Time Radio Systems Ltd",
    author_email="jack@realtimeradio.co.uk",
    description="A correlator / beamformer pipeline for the LWA-352",
    scripts=glob.glob('scripts/*.py'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/realtimeradio/caltech-bifrost-dsp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu 18.04",
    ],
    python_requires='>=3.5',
    install_requires=install_requires,
)

