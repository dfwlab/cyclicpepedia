# CyclicPepedia: A knowledge base of natural and synthetic cyclic peptides

Lei Liu<sup>1, #</sup>, Liu Yang<sup>2, #</sup>, Suqi Cao<sup>2</sup>, Zhigang Gao<sup>3</sup>, Bin Yang<sup>4</sup>, Guoqing Zhang<sup>5, *</sup>, Ruixin Zhu<sup>1, *</sup>, Dingfeng Wu<sup>2, *</sup>

### Web crawling
The <code>/Web crawling</code> directory stores the crawler codes of different data source. 

### Environment
Build a Docker image through <code>dockerfile</code> or pull docker image through this link:<code>docker pull registry.cn-hangzhou.aliyuncs.com/dfwlab/cyclicpepedia:20240106</code>. You can use this image with the following command like: <code>docker run -it -p 80:8888 -v $(PWD):/Cyclicpepedia $imageName:$tag /bin/bash</code>. And then install the Jupyterlab by <code>pip install jupyterlab</code> and lanch it by <code>jupyter lab --allow-root --ip=0.0.0.0 --port=8888</code>.

### Dataset
The <code>/Dataset</code> directory stores the main data resources of the <a href="https://www.biosino.org/iMAC/cyclicpepedia/">CyclicPepedia</a>, and you can also download these data from the <a href="https://www.biosino.org/iMAC/cyclicpepedia/download">Downloads</a> of CyclicPepedia.

### Tools
Codes in <code>/Tools</code> for the cyclic peptide tool used by CyclicPepedia and its test notebooks. You need to ensure that the program is valid in the same environment (build a Docker image through dockerfile and <code>pip install jupyterlab</code>), or pull docker image through this link:<code>docker pull registry.cn-hangzhou.aliyuncs.com/dfwlab/cyclicpepedia:20240106</code>. 

* Struc2Seq (structure2sequence.py) and Test_Struc2seq.ipynb
* Seq2Struc (sequence2structure.py) and Test_Seq2struc.ipynb
* ga (graph_alignment.py) and Test_GraphAlignment.ipynb
* pp (peptide_properties.py) and Test_Peptide_properties.ipynb
* RDKit_format_issue.ipynb
