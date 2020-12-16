# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.append(os.path.abspath('../pipeline'))
from lwa352_pipeline.blocks.block_base import Block
from lwa352_pipeline.blocks.corr_block import Corr
from lwa352_pipeline.blocks.dummy_source_block import DummySource
from lwa352_pipeline.blocks.corr_acc_block import CorrAcc
from lwa352_pipeline.blocks.corr_subsel_block import CorrSubsel
from lwa352_pipeline.blocks.corr_output_full_block import CorrOutputFull
from lwa352_pipeline.blocks.corr_output_part_block import CorrOutputPart
from lwa352_pipeline.blocks.copy_block import Copy
from lwa352_pipeline.blocks.capture_block import Capture
from lwa352_pipeline.blocks.beamform_block import Beamform
from lwa352_pipeline.blocks.beamform_sum_block import BeamformSum
from lwa352_pipeline.blocks.beamform_sum_single_beam_block import BeamformSumSingleBeam
from lwa352_pipeline.blocks.beamform_sum_beams_block import BeamformSumBeams
from lwa352_pipeline.blocks.beamform_vlbi_output_block import BeamformVlbiOutput
from lwa352_pipeline.blocks.beamform_vacc_block import BeamVacc
from lwa352_pipeline.blocks.beamform_output_block import BeamformOutputBf as BeamformOutput
from lwa352_pipeline.blocks.triggered_dump_block import TriggeredDump
####sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'LWA Correlator'
copyright = '2020, Jack Hickish'
author = 'Jack Hickish'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinxcontrib.programoutput',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

numfig=True
