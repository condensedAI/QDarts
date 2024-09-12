# Minimal makefile for Sphinx documentation
#
# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXSOURCEDIR     = sphinx
BUILDDIR      = build
PYTHONBUILD ?= python3 -m build

.PHONY: doc build

doc:
	@$(SPHINXBUILD) -M html "$(SPHINXSOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	cp -r $(BUILDDIR)/html/* docs/
build:
	@$(PYTHONBUILD)
