# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QDarts'
copyright = '2024, Oswin Krause, Jan Krzywda, Weikun Liu, Evert van Nieuwenburg'
author = 'Oswin Krause, Jan Krzywda, Weikun Liu, Evert van Nieuwenburg'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon'
    ]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


# -- Package options -------

import sys
import os
sys.path.insert(0, os.path.abspath('../src'))
autosummary_generate = True
autosummary_imported_members = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True
}
