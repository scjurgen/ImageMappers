#!/usr/bin/env bash
$(brew --prefix)/bin/python3.13 -m venv .venv
source .venv/bin/activate
history | grep pip | grep require
pip install -v -r requirements.txt
