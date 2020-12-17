import setuptools
import glob
import os

VERSION = "1.0.0"

desc_file = "README.md"
long_description = ""
if os.path.isfile(desc_file):
    with open(desc_file, "r") as fh:
        long_description = fh.read()

setuptools.setup(
    name="lwa352-pipeline-control",
    version=VERSION,
    author="Real-Time Radio Systems Ltd",
    author_email="jack@realtimeradio.co.uk",
    description="A control library for the LWA-352 correlator/beamformer pipeline",
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
)

