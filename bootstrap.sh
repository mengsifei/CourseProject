#!/bin/sh
apt update
apt install python cmake
apt install pip
pip install --no-cache-dir -r requirements.txt