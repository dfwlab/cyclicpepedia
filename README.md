# CyclicPepedia: A knowledge base of natural and synthetic cyclic peptides

Lei Liu<sub>1, #</sub>, Liu Yang2, #, Suqi Cao2, Zhigang Gao3, Bin Yang4, Guoqing Zhang5, *, Ruixin Zhu1, *, Dingfeng Wu2, *


### Environment
Build a Docker image through <code>dockerfile</code> or pull docker image through this link:<code>docker pull registry.cn-hangzhou.aliyuncs.com/dfwlab/cyclicpepedia:20240106</code>.

### Dataset
The <code>/Dataset</code> directory stores the main data resources of the <a href="https://www.biosino.org/iMAC/cyclicpepedia/">CyclicPepedia</a>, and you can also download these data from the <a href="https://www.biosino.org/iMAC/cyclicpepedia/download">Downloads</a> of CyclicPepedia.

### Tools
Codes in <code>/Tools</code> for the cyclic peptide tool used by CyclicPepedia and its test notebooks. You need to ensure that the program is valid in the same environment (build a Docker image through dockerfile and <code>pip install Jupyterlab</code>), or pull docker image through this link:<code>docker pull registry.cn-hangzhou.aliyuncs.com/dfwlab/cyclicpepedia:20240106</code>. You can use this image with the following command like: <code>docker run -it -p 80:8000 -v $(PWD):/Cyclicpepedia $imageName:$tag /bin/bash</code>

* Struc2Seq (structure2sequence.py) and Test_Struc2seq.ipynb
* Seq2Struc (sequence2structure.py) and Test_Seq2struc.ipynb
* ga (graph_alignment.py) and Test_GraphAlignment.ipynb
* pp (peptide_properties.py) and Test_Peptide_properties.ipynb
* RDKit_format_issue.ipynb
