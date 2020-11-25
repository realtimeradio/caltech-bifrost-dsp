.. |repopath| replace:: https://github.com/realtimeradio/caltech-bifrost-dsp
.. |ibv-version| replace:: 4.9 LTS
.. |py-version| replace:: >=3.5

Installation
============

The LWA 352 Correlator/Beamformer pipeline is available at |repopath|.
Follow the following instructions to download and install the pipeline.

Specify the build directory by defining the ``BUILDDIR`` environment variable, eg:

.. code-block::

  export BUILDDIR=~/src/
  mkdir -p $BUILDDIR

Get the Source Code
-------------------

Clone the repository and its dependencies with:

.. code-block::

  # Clone the main repository
  cd $BUILDDIR
  git clone https://github.com/realtimeradio/caltech-bifrost-dsp
  # Clone relevant submodules
  cd caltech-bifrost-dsp
  git submodule init
  git submodule update

Install Prerequisites
---------------------

The following libraries should be installed via the Ubuntu package manager:

.. code-block::

  apt install exuberant-ctags build-essential autoconf libtool exuberant-ctags libhwloc-dev python3-venv

The following 3rd party libraries must also be obtained and installed:

CUDA
~~~~

CUDA can be installed as follows:

.. code-block::

  # Make a directory for the cuda source
  mkdir -p $BUILDDIR/cuda
  cd $BUILDDIR/cuda
  # Download the CUDA installer
  wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
  
  # blacklist nouveau drivers before installing nvidia drivers
  sudo su
  echo "blacklist nouveau" > /etc/modprobe.d/blacklist-nouveau.conf
  echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf
  update-initramfs -u
  exit
  
  # reboot the machine and nouveau drivers [hopefully] won't start
  # reboot now

After rebooting, install the CUDA libraries

.. code-block::

  cd $BUILDDIR/cuda
  sudo sh cuda_11.0.2_450.51.05_linux.run
  # Add CUDA executables to $PATH
  echo "export PATH=/usr/local/cuda/bin:${PATH}" >> ~/.bashrc
  source ~/.bashrc

This CUDA install script will take a minute to unzip and run the installer.
If it fails, log messages can be found in ``/var/log/nvidia-installer.log`` and ``/var/log/cuda-installer.log``.

IB Verbs
~~~~~~~~

The LWA pipeline uses Infiniband Verbs for fast UDP packet capture.
The recommended version is |ibv-version|.
This can be obtained from https://www.mellanox.com/support/mlnx-ofed-matrix?mtag=linux_sw_drivers

xGPU
~~~~

xGPU is submoduled in the main pipeline repository, to ensure version compatibility. Install with:

.. code-block::

  cd $BUILDDIR/caltech-bifrost-dsp
  ./install_xgpu

Bifrost
~~~~~~~

Bifrost is submoduled in the main pipeline repository, to ensure version compatibility.

The version provided requires Python |py-version|.
It is recommended that the bifrost package is installed within a Python version environment.

To install bifrost:

.. code-block::

  cd $BUILDDIR/caltech-bifrost-dsp/bifrost
  make
  make install


Install the Pipeline
--------------------

After installing the prerequisites above, the LWA pipeline can be installed with

.. code-block::

  cd $BUILDDIR/caltech-lwa-dsp/pipeline
  # Be sure to run the installation in the
  # appropriate python environment!
  python setup.py install
