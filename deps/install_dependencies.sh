#!/usr/bin/env bash

# MG5aMC
wget https://launchpad.net/mg5amcnlo/3.0/3.5.x/+download/MG5_aMC_v3.5.8.tar.gz
tar -xzf MG5_aMC_v3.5.8.tar.gz

# MadSymbolic
cd MG5_aMC_v3_5_8/PLUGIN
git clone git@github.com:alphal00p/madsymbolic.git
cd -

# GammaLoop
git clone -b dev git@github.com:alphal00p/gammaloop.git
cd gammaloop
git checkout 9d2303acf2e677cd8aa4236ef73808f830fc7ffd
./bin/compile.sh
cd -

# Symbolica community
#git clone git@github.com:benruijl/symbolica-community.git
git clone git@github.com:alphal00p/symbolica-community.git
cd symbolica-community
git checkout 512a9d72c8fa33a419c67ed7746f5b7809d60aef
python3 -m venv .venv
maturin develop --release
cd -

# pyhton dependencies
python3 -m pip install -r ../madloop7/requirements.txt
