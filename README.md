# Machine Learning

## Installation Instruction

### IPython Notebook

Install IPython:

    sudo apt-get install build-essential python-dev python-numpy \
    python-numpy-dev python-scipy libatlas-dev g++ python-matplotlib \
    ipython
    
    sudo apt-get install ipython-notebook python-matplotlib
    
    sudo apt-get install python-pip
    
    pip install -U scikit-learn

Run the notebook:

    cd <notebook folder>
    
To run locally:

    ipython notebook <optional notebook file>
    
It will open the notebook in a browser.

To access the notebook from a remote machine browser. First `ssh` to the notebook computer and run command:

    ipython notebook --no-browser --port=7000
    
On the machine you are accessing the notebook from. Run: 

    ssh -N -f -L localhost:6025:localhost:7000 username@<ip of machine running notebook>
    
Browse following URL on remote machine:

    http://locahost:6025

Add following in the notebook to show graphs inline:

    %matplotlib inline
    
Add followings command to show graphs inline without scrollbar:

    %%javascript
    IPython.OutputArea.auto_scroll_threshold = 9999;

Convert notebook to html by:

    $ ipython nbconvert --to FORMAT notebook.ipynb
