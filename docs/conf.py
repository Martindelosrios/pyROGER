import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyROGER'
copyright = '2025, Martin de los Rios'
author = 'Martin de los Rios'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",       # Documentar desde docstrings
    "sphinx.ext.napoleon",      # Soporte Google/Numpy style
    "sphinx.ext.viewcode",      # Links al código fuente
    "sphinx_autodoc_typehints", # Anotaciones de tipos
]



templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "ignore-module-all": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
