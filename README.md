# Machine Learning

## Installation Instruction

### IPython Notebook

Install IPython:

    sudo apt-get install build-essential python-dev python-numpy \
    python-numpy-dev python-scipy libatlas-dev g++ python-matplotlib \
    ipython
    
    sudo apt-get install ipython-notebook python-matplotlib

Run the notebook:

    cd <notebook folder>
    ipython notebook --no-browser --port=7000
    
To access Notebook from a remote machine:

    ssh -N -f -L localhost:6025:localhost:7000 username@<ip of notebook pc>
    
Browse following URL on remote machine:

    http://locahost:6025

Add following in the notebook to show graphs inline:

    %matplotlib inline
    
Add followings command to show graphs inline without scrollbar:

    %%javascript
    IPython.OutputArea.auto_scroll_threshold = 9999;

Convert notebook to html by:

    $ ipython nbconvert --to FORMAT notebook.ipynb
