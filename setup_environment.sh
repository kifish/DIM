~/anaconda3/bin/conda create -n MRFN python=2.7
source ~/anaconda3/bin/activate
conda activate MRFN
conda install -c anaconda tensorflow-gpu=1.4
# cudnn-7.1.3  
# cudatoolkit-8.0
# tensorflow-gpu            1.4.1                         0    anaconda
# tensorflow-gpu-base       1.4.1            py27h01caf0a_0    anaconda
# tensorflow-tensorboard    1.5.1            py27hf484d3e_1    anaconda
# cudnn-7.1.3 这个版本过高
conda install cudnn=7.0.5
# cudnn-7.0.5  
conda install gensim
pip install keras==2.0.8
pip install tqdm






pip install nltk