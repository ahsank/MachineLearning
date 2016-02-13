# Machine Learning

## Installation Instruction

### IPython Notebook

Install IPython:

    sudo apt-get install build-essential python-dev python-numpy \
    python-numpy-dev python-scipy libatlas-dev g++ python-matplotlib \
    ipython

Run the notebook:

    cd <notebook folder>
    ipython notebook --no-browser --port=7000
    
To access Notebook from a remote machine:

    ssh -N -f -L localhost:6025:localhost:7000 username@<ip of notebook pc>
    
Browse following URL on remote machine:

    http://locahost:6025

Add following to show graphs:

    %matplotlib inline
    
    
