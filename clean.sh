git pull origin rci-benchmarking 
git submodule update 
./install_xgpu 
cd bifrost 
make clean 
make 
make install 
cd ../pipeline 
python setup.py install

