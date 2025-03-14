#!/usr/bin/env bash

# MG5aMC
wget https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.5.7.tar.gz
tar -czf MG5_aMC_v3.5.7.tar.gz

# MadSymbolic
cd MG5_aMC_v3_5_7/PLUGIN
git clone git@github.com:alphal00p/madsymbolic.git
cd -

# GammaLoop
git clone git@github.com:alphal00p/gammaloop.git
cd gammaloop
./bin/compile.sh
cd -

# Symbolica community
git clone git@github.com:benruijl/symbolica-community.git
cd symbolica-community
python3 -m venv .venv
maturin develop --release
cd -

# pyhton dependencies
python3 -m pip install -r ../madloop7/requirements.txt
